# Development Log

## Milestone 1: Feasibility MVP Completed

Date: [today's date]

Implemented a Flutter-based mobile prototype connected to a FastAPI backend. The system accepts manually entered URLs and QR-scanned URLs, sends them to the backend, and displays a risk score, risk level, color-coded warning, plain-English explanation, and recommended action.

Current backend uses rule-based dummy logic. This will later be replaced by a machine learning model after the literature review and dataset preparation phases.

Completed components:
- Flutter mobile interface
- Manual URL input
- QR scanner screen
- FastAPI prediction endpoint
- Risk score response
- Color-coded result display
- Explanation and recommendation output

Next step:
Begin systematic literature review protocol and identify ML features for the real phishing detection model.