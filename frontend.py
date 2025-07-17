# frontend.py
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import time

# --- CONSTANTS ---
API_BASE_URL = "http://localhost:8000/api"
ONLY_CANDIDATES = "You must be a candidate to perform this action."
ONLY_EMPLOYERS = "You must be an employer to perform this action."
ONLY_ADMINS = "You must be an admin to perform this action."
PROFILE_FIRST = "Please create your candidate profile first!"
UPLOAD_RESUME_FIRST = "Please upload your resume first!"
STATUS_LIST = ["applied", "reviewed", "interviewed", "hired", "rejected"]

st.set_page_config(
    page_title="AI Recruiting Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
## FIX: Ensured all necessary session state keys are initialized to prevent errors.
for key in ['token', 'user_type', 'user_info', 'delete_confirm_user_id', 'open_user_expander_id', 'viewing_job_applications', 'viewing_ai_recommendations', 'open_job_expander_id', 'current_job_to_apply', 'show_apply_form']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- HELPERS ---
def sort_by_key(items, key, reverse=True):
    return sorted(items, key=lambda x: x[key], reverse=reverse)

def show_user_info(user_info):
    st.write(f"**Name:** {user_info.get('full_name', 'N/A')}")
    st.write(f"**Email:** {user_info.get('email', 'N/A')}")

## FIX: Corrected show_candidate_profile to handle an empty skills list.
def show_candidate_profile(profile):
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.write(f"**Experience:** {profile.get('experience_years', 0)} years")
        st.write(f"**Location:** {profile.get('location', 'N/A')}")
        st.write(f"**Education:** {profile.get('education', 'N/A')}")
    with profile_col2:
        # This line is changed to prevent the TypeError
        skills_list = profile.get('skills') or []
        st.write(f"**Skills:** {', '.join(skills_list)}")
        st.write(f"**Salary Expectation:** ${profile.get('salary_expectation', 0):,}")
        st.write(f"**Resume:** {profile.get('resume_filename', 'Not uploaded')}")

def show_job_recommendation(rec, i):
    with st.container(border=True):
        job = rec['job']
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.markdown(f"#### {job['title']}")
            st.write(f"**🏢 Company:** {job['company_name']}")
            st.write(f"**📍 Location:** {job['location']}")
        
        with c2:
            st.metric("AI Match Score", f"{rec['match_score']:.1f}%")

        with c3:
            # When this button is clicked, it will set session state to show the application form.
            if st.button("🚀 Apply Now", key=f"apply_{i}", type="primary"):
                st.session_state['current_job_to_apply'] = job['id']
                st.session_state['show_apply_form'] = True
                st.rerun()

        with st.expander("View AI Analysis and Details"):
            st.markdown(f"**🤖 AI Recommendation Reason:** *{rec['recommendation_reason']}*")
            st.markdown(f"**✅ Matched Skills:** `{'`, `'.join(rec['matched_skills'])}`")
            if rec['missing_skills']:
                st.markdown(f"**❌ Missing Skills:** `{'`, `'.join(rec['missing_skills'])}`")

def show_application_expander(app):
    applied_at_str = datetime.fromisoformat(app['applied_at']).strftime('%B %d, %Y')
    with st.expander(f"**{app['job_title']}** - Status: {app['status'].title()} (Applied on {applied_at_str})"):
        st.metric("Your Match Score", f"{app['match_score']:.1f}%" if app['match_score'] is not None else "N/A")
        
        # This block is changed to use st.markdown
        if app.get("ai_report"):
            st.write("**AI Assessment of Your Profile for this Job:**")
            st.markdown(app["ai_report"], unsafe_allow_html=True)

def show_api_error(response):
    try:
        error_detail = response.json().get("detail", "Unknown error")
    except json.JSONDecodeError:
        error_detail = response.text
    st.error(f"API Error: {response.status_code} - {error_detail}")

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

## FIX: Refactored API request logic for better error handling and auth flow.
## FIX: The entire make_api_request function is updated to correctly handle the 204 status code.
def make_api_request(endpoint, method="GET", data=None, files=None, params=None):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, headers=headers, json=data, files=files, params=params)
        
        # Success codes that are expected to have a JSON body
        if response.status_code in [200, 201]:
            return response.json()
        
        # Success code for DELETE requests, which has no body
        if response.status_code == 204:
            return True  # Return a simple success indicator
        
        # Authentication-related errors
        if response.status_code in (401, 403):
            st.error("Session expired or unauthorized. Please log in again.")
            logout()
            return None
            
        # All other non-successful status codes
        else:
            show_api_error(response)
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Is it running?")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- DASHBOARDS & PAGES ---

def admin_dashboard():
    st.title("🛡️ Admin Dashboard")
    with st.sidebar:
        st.header(f"Admin: {st.session_state.user_info['full_name']}")
        if st.button("Logout"):
            logout()

    st.subheader("User Management")
    users_data = make_api_request("/admin/users")
    if users_data and users_data.get("users"):
        df = pd.DataFrame(users_data["users"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        for user in users_data["users"]:
            if user['id'] == st.session_state.user_info['id']: continue # Admin can't delete self
            
            with st.expander(f"Manage: {user['full_name']} ({user['email']})"):
                if st.session_state.delete_confirm_user_id == user['id']:
                    st.warning(f"Are you sure you want to delete {user['email']}? This is irreversible.")
                    col1, col2 = st.columns(2)
                    if col1.button("✅ Confirm Delete", key=f"confirm_delete_{user['id']}", type="primary"):
                        if make_api_request(f"/admin/users/{user['id']}", method="DELETE"):
                            st.success("User deleted.")
                            st.session_state.delete_confirm_user_id = None
                            time.sleep(1)
                            st.rerun()
                    if col2.button("❌ Cancel", key=f"cancel_delete_{user['id']}"):
                        st.session_state.delete_confirm_user_id = None
                        st.rerun()
                else:
                    if st.button("Delete User", key=f"delete_{user['id']}"):
                        st.session_state.delete_confirm_user_id = user['id']
                        st.rerun()

def login_page():
    st.title("🤖 AI Recruiting Platform")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                data = {"email": email, "password": password}
                result = make_api_request("/auth/login", method="POST", data=data)
                if result:
                    st.session_state.token = result["access_token"]
                    st.session_state.user_type = result["user_type"]
                    st.rerun()
    with tab2:
        with st.form("register_form"):
            user_type = st.selectbox("I am a...", ["candidate", "employer"])
            full_name = st.text_input("Full Name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            if st.form_submit_button("Register"):
                data = {"email": email, "password": password, "full_name": full_name, "user_type": user_type}
                result = make_api_request("/auth/register", method="POST", data=data)
                if result:
                    st.success("Registration successful! Logging you in...")
                    st.session_state.token = result["access_token"]
                    st.session_state.user_type = result["user_type"]
                    time.sleep(1)
                    st.rerun()

## FIX: The entire candidate_dashboard is updated for a smoother workflow.
def candidate_dashboard():
    st.title("👤 Candidate Dashboard")
    with st.sidebar:
        st.header(st.session_state.user_info.get('full_name', 'Candidate'))
        if st.button("Logout"):
            logout()

    profile_data = make_api_request("/user/profile")
    if not profile_data: return 
    candidate_profile = profile_data.get("profile")

    # --- Profile Section ---
    st.subheader("Your Profile")
    if not candidate_profile:
        st.info(PROFILE_FIRST)
        with st.expander("Create Profile", expanded=True):
            create_candidate_profile()
    else:
        show_candidate_profile(candidate_profile)
        resume_exists = candidate_profile.get('resume_filename')
        with st.expander("Upload/Update Resume", expanded=not resume_exists):
            if not resume_exists:
                st.info("Your profile is created! Now, upload your resume to get AI-powered job recommendations.")
            upload_resume()

    st.divider()

    # --- Application Form Section ---
    # This section will only appear when the "Apply" button is clicked.
    if st.session_state.get('show_apply_form'):
        st.subheader("📝 Submit Your Application")
        job_id_to_apply = st.session_state.get('current_job_to_apply')
        if job_id_to_apply:
            apply_for_job(job_id_to_apply)
        st.divider()

    # --- Recommendations Section ---
    st.subheader("🎯 AI-Powered Job Recommendations")
    if candidate_profile and candidate_profile.get('skills'):
        recs_data = make_api_request("/candidate/job-recommendations")
        if recs_data and recs_data.get("recommendations"):
            for i, rec in enumerate(recs_data["recommendations"]):
                show_job_recommendation(rec, i)
        else:
            st.info("No recommendations found. More jobs may be added soon!")
    else:
        st.warning(UPLOAD_RESUME_FIRST)
    
    st.divider()

    # --- My Applications Section ---
    st.subheader("📄 My Applications")
    my_apps_data = make_api_request("/candidate/applications")
    if my_apps_data and my_apps_data.get("applications"):
        for app in my_apps_data["applications"]:
            show_application_expander(app)
    else:
        st.info("You have not applied to any jobs yet.")
        
def employer_dashboard():
    st.title("🏢 Employer Dashboard")
    with st.sidebar:
        st.header(st.session_state.user_info['full_name'])
        if st.button("Logout"):
            logout()

    analytics = make_api_request("/analytics/dashboard")
    if analytics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Jobs Posted", analytics.get("jobs_count", 0))
        c2.metric("Active Jobs", analytics.get("active_jobs_count", 0))
        c3.metric("Total Applications Received", analytics.get("total_applications", 0))

    with st.expander("➕ Create New Job Posting"):
        create_job()
    
    st.subheader("📋 Your Job Postings")
    jobs_data = make_api_request("/jobs")
    if jobs_data and jobs_data.get("jobs"):
        for job in jobs_data["jobs"]:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"#### {job['title']}")
                c1.write(f"Applications: {job['applications_count']} | Status: {'🟢 Active' if job['is_active'] else '🔴 Inactive'}")
                if c2.button("View Details & Applicants", key=f"view_{job['id']}"):
                    st.session_state.viewing_job_applications = job['id']
    else:
        st.info("No jobs posted yet.")

    if st.session_state.viewing_job_applications:
        view_job_applicants(st.session_state.viewing_job_applications)

## FIX: Corrected the view_job_applicants function to remove the invalid 'key' argument.
def view_job_applicants(job_id):
    st.divider()
    st.header("Applicant Details")

    applications_data = make_api_request(f"/jobs/{job_id}/applications")
    recommendations_data = make_api_request(f"/employer/job/{job_id}/candidate-recommendations")

    app_tab, rec_tab = st.tabs(["Submitted Applications", "AI Candidate Recommendations"])

    with app_tab:
        if applications_data and applications_data.get("applications"):
            st.subheader("Ranked Applications for this Job")
            for app in applications_data["applications"]:
                # The 'key' argument has been removed from the line below.
                with st.expander(f"🧑‍💻 {app['candidate_name']} (🎯 {app['match_score']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    c1.metric("Skill Match", f"{app['skill_match_percentage']:.1f}%")
                    c2.metric("Experience Match", f"{app['experience_match_percentage']:.1f}%")
                    new_status = st.selectbox("Update Status", STATUS_LIST, index=STATUS_LIST.index(app['status']), key=f"status_{app['id']}")
                    if st.button("Save Status", key=f"save_{app['id']}"):
                        update_application_status(app['id'], new_status)
                    if app.get("ai_report"):
                        st.write("**AI Assessment Report**")
                        st.markdown(app["ai_report"], unsafe_allow_html=True)
        else:
            st.info("No applications received for this job yet.")
            
    with rec_tab:
        if recommendations_data and recommendations_data.get("recommendations"):
            st.subheader(f"AI-Sourced Candidates for: {recommendations_data['job_title']}")
            for rec in recommendations_data["recommendations"]:
                if rec['has_applied']: continue 
                with st.container(border=True):
                    c1, c2 = st.columns([3,1])
                    c1.markdown(f"#### 🧑‍💻 {rec['candidate_name']}")
                    c1.write(f"Experience: {rec['experience_years']} years | Location: {rec.get('location', 'N/A')}")
                    c2.metric("AI Match Score", f"{rec['match_score']:.1f}%")
                    with st.expander("View AI Analysis"):
                        st.write(f"**Matched Skills:** {', '.join(rec['matched_skills'])}")
                        st.write(f"**Missing Skills:** {', '.join(rec['missing_skills'])}")
                        if st.button("Invite to Apply", key=f"invite_{rec['candidate_id']}"):
                            st.success(f"Invitation sent to {rec['candidate_name']}!")
        else:
            st.info("No AI recommendations available for this job yet.")

# --- FORMS & ACTIONS ---

## FIX: Added the missing "Salary Expectation" input field to the form.
def create_candidate_profile():
    with st.form("candidate_profile_form"):
        full_name = st.text_input("Full Name")
        experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
        education = st.text_area("Highest Education")
        location = st.text_input("Current Location (e.g., New York, NY)")
        
        # This is the new input field for salary expectation.
        salary_expectation = st.number_input(
            "Desired Annual Salary ($)", 
            min_value=0, 
            step=1000, 
            help="Enter 0 if you prefer not to say."
        )

        if st.form_submit_button("Create Profile"):
            # The data payload is updated to include the new field.
            data = {
                "full_name": full_name, 
                "experience_years": experience_years, 
                "education": education, 
                "location": location,
                # It will be saved as NULL if the user enters 0.
                "salary_expectation": salary_expectation if salary_expectation > 0 else None
            }

            if make_api_request("/candidate/profile", method="POST", data=data):
                st.success("Profile created! Now please upload your resume.")
                time.sleep(1)
                st.rerun()

def upload_resume():
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=['pdf', 'docx'])
    if uploaded_file and st.button("Process Resume"):
        with st.spinner("Analyzing your resume..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            if make_api_request("/candidate/upload-resume", method="POST", files=files):
                st.success("Resume processed successfully! Your recommendations have been updated.")
                time.sleep(1)
                st.rerun()

## FIX: Added min and max salary fields to the job creation form.
def create_job():
    with st.form("job_form", border=False):
        title = st.text_input("Job Title")
        company_name = st.text_input("Company Name")
        location = st.text_input("Location")
        description = st.text_area("Job Description", height=150)
        required_skills = st.text_input("Required Skills (comma-separated)")
        min_experience = st.number_input("Minimum Years of Experience", min_value=0, step=1)
        
        # New input fields for salary range.
        col1, col2 = st.columns(2)
        with col1:
            salary_min = st.number_input(
                "Minimum Annual Salary ($)", 
                min_value=0, 
                step=1000,
                help="Enter 0 if there is no minimum."
            )
        with col2:
            salary_max = st.number_input(
                "Maximum Annual Salary ($)", 
                min_value=0, 
                step=1000,
                help="Enter 0 if there is no maximum."
            )

        if st.form_submit_button("Create Job", type="primary"):
            # Updated data payload to include the new salary fields.
            data = {
                "title": title, 
                "description": description, 
                "company_name": company_name, 
                "location": location,
                "required_skills": [s.strip() for s in required_skills.split(",")],
                "min_experience": min_experience,
                "education_requirement": "Not specified", # Default value
                "job_type": "full-time", # Default value
                "salary_range_min": salary_min if salary_min > 0 else None,
                "salary_range_max": salary_max if salary_max > 0 else None
            }
            if make_api_request("/jobs", method="POST", data=data):
                st.success("Job created successfully!")
                time.sleep(1)
                st.rerun()
                
def apply_for_job(job_id):
    with st.form(f"apply_form_{job_id}"):
        st.info(f"You are applying for the job shown above.")
        cover_letter = st.text_area("Cover Letter (Optional)")
        
        if st.form_submit_button("Confirm & Submit Application", type="primary"):
            with st.spinner("Submitting your application..."):
                data = {"job_id": job_id, "cover_letter": cover_letter}
                result = make_api_request("/applications", method="POST", data=data)
                if result:
                    st.success("Application submitted successfully!")
                    # Clear the form flags and rerun to refresh the page
                    st.session_state['show_apply_form'] = False
                    st.session_state['current_job_to_apply'] = None
                    time.sleep(1)
                    st.rerun()
                    
def update_application_status(application_id, status):
    if make_api_request(f"/applications/{application_id}/status", method="PUT", data={"status": status}):
        st.success("Status updated!")
        time.sleep(1)
        st.rerun()

# --- MAIN APP ROUTER ---

def main():
    if not st.session_state.token:
        login_page()
    else:
        if not st.session_state.user_info:
            st.session_state.user_info = make_api_request("/user/profile").get('user')
            if not st.session_state.user_info: return

        if st.session_state.user_type == "candidate":
            candidate_dashboard()
        elif st.session_state.user_type == "employer":
            employer_dashboard()
        elif st.session_state.user_type == "admin":
            admin_dashboard()

if __name__ == "__main__":
    main()
