"""Visual styles for the Streamlit knowledge assistant."""

APP_CSS = """
<style>
html, body, [class*="css"], .stApp {
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
}
.stApp { background: #f6f9f8; color: #1d3438; }
.block-container { max-width: 1050px; padding-top: 2rem; }
[data-testid="stSidebar"] { background: #eef6f2; border-right: 1px solid #deebe6; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.side-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.brand-icon {
    display: grid; place-items: center; width: 40px; height: 40px;
    border-radius: 13px; background: #137d70; color: white; font-size: 24px;
}
.side-brand strong { display: block; font-size: 18px; color: #173f3e; }
.side-brand small { display: block; font-size: 11px; color: #78908a; }
.hero {
    position: relative; overflow: hidden; padding: 32px 38px; border-radius: 23px;
    background: radial-gradient(circle at 88% 12%, #277f75 0, transparent 34%),
                linear-gradient(112deg, #123940, #126358);
    color: white; box-shadow: 0 17px 34px rgba(18, 72, 65, .14);
}
.hero:after {
    content: ''; position: absolute; width: 230px; height: 230px; right: -50px; bottom: -120px;
    border: 1px solid rgba(255,255,255,.18); border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(255,255,255,.04), 0 0 0 57px rgba(255,255,255,.025);
}
.hero-label { color: #a8e6d1; font-size: 12px; font-weight: 700; letter-spacing: 2px; }
.hero h1 { color: white; font-size: 34px; line-height: 1.35; margin: 14px 0 8px; }
.hero p { color: #d2e9e2; font-size: 14px; margin: 0; }
.info-card {
    display: flex; flex-direction: column; gap: 6px; min-height: 83px;
    margin: 17px 0 27px; padding: 17px 18px; border: 1px solid #e0ece7;
    border-radius: 15px; background: white; box-shadow: 0 5px 15px rgba(26, 78, 70, .035);
}
.info-card b { color: #17685f; font-size: 14px; }
.info-card span { color: #728980; font-size: 12px; }
.chat-title { display: flex; align-items: baseline; gap: 12px; margin: 3px 0 17px; font-size: 20px; font-weight: 700; }
.chat-title span { color: #81938d; font-size: 12px; font-weight: 400; }
.welcome-card {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 190px; border: 1px dashed #cadfd5; border-radius: 18px;
    background: #fcfefd; text-align: center; color: #52776d;
}
.welcome-card span { color: #1c9a82; font-size: 28px; }
.welcome-card strong { margin-top: 8px; font-size: 17px; color: #254841; }
.welcome-card p { margin: 7px 16px 0; font-size: 12px; }
[data-testid="stChatMessage"] { background: white; border: 1px solid #e4eeea; border-radius: 15px; }
[data-testid="stChatInput"] { border-color: #b4d6ca; }
@media (max-width: 700px) {
    .block-container { padding-top: 1rem; }
    .hero { padding: 24px 21px; border-radius: 18px; }
    .hero h1 { font-size: 26px; }
    .info-card { margin: 7px 0; min-height: 65px; }
}
</style>
"""
