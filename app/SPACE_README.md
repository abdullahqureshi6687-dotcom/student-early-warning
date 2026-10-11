---
title: Student Early-Warning System
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predicts students at risk of failing before final grades
---

# Student Early-Warning System

Predicts which students are at risk of failing (final grade G3 < 10) using
background information and the first-period grade G1, early enough to intervene.

- Model: Logistic Regression (class-balanced) in a scikit-learn Pipeline
- Data: UCI Student Performance dataset (Portuguese course, 649 students)
- Shows a fail-risk score, a Low / Medium / High risk band, and the top 3 reasons

Full code, analysis and results: [GitHub repository](https://github.com/abdullahqureshi6687-dotcom/student-early-warning)