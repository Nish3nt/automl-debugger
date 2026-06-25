"""
AutoML Debugger — Professional Report Generator  (Milestone 3 v4.0)
====================================================================
Feature 15: Multi-Section Groq Report

6 sections, each written as real professional paragraphs for a
specific audience — not bullet points, not templates.

  1. Executive Summary        — for managers (plain English, verdict)
  2. Dataset Quality Analysis — for ML engineers (technical detail)
  3. Risk Assessment          — for compliance (what can go wrong)
  4. Feature Analysis         — for data scientists (what to transform)
  5. Recommended Next Steps   — for project planning (exact actions)
  6. Model Recommendations    — for the team (which algorithm, why)

PDF is structured as a professional audit document with cover page,
scorecard table, Groq sections, and data tables.
"""

from __future__ import annotations

import io
import os
import json
from datetime import datetime
from typing import Any


# ─────────────────────────────────────────────────────────────────
# GROQ MULTI-SECTION REPORT WRITER
# ─────────────────────────────────────────────────────────────────

def generate_groq_report(
    profile:          dict,
    health:           dict,
    scorecard:        dict,
    leakage:          dict,
    redundancy:       dict,
    missing_pattern:  dict,
    distributions:    dict,
    type_inference:   dict,
    sample_check:     dict,
    ts_info:          dict,
    target_column:    str,
    task_type:        str,
    leakage_prob:     dict | None = None,
    outlier_rca:      dict | None = None,
    drift_sim:        dict | None = None,
    fe_roadmap:       dict | None = None,
    api_key:          str | None = None,
) -> dict[str, str]:
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    # Build rich context for Groq
    context = {
        "dataset": {
            "rows":         profile.get("rows", 0),
            "columns":      profile.get("columns", 0),
            "numeric":      profile.get("numeric_features", 0),
            "categorical":  profile.get("categorical_features", 0),
            "missing_pct":  profile.get("missing_pct", 0),
            "duplicates":   profile.get("duplicate_rows", 0),
            "target":       target_column,
            "task":         task_type,
        },
        "quality": {
            "grade":      scorecard.get("overall_grade", "?"),
            "score":      scorecard.get("overall_score", 0),
            "verdict":    scorecard.get("overall_verdict", ""),
            "health":     health.get("total", 0),
            "dimensions": {k: {"score": v["score"], "reason": v["reason"]}
                           for k, v in health.get("dimensions", {}).items()},
        },
        "risks": {
            "leakage_candidates": leakage.get("leakage_candidates", []),
            "leakage_prob_critical": (leakage_prob or {}).get("n_critical", 0),
            "leakage_prob_high":     (leakage_prob or {}).get("n_high", 0),
            "timeseries":            ts_info.get("is_timeseries", False),
            "ts_frequency":          ts_info.get("frequency_guess", ""),
            "drift_risk":            (drift_sim or {}).get("risk", "UNKNOWN"),
            "drift_pct":             (drift_sim or {}).get("drift_pct", 0),
            "sample_issues":         sample_check.get("issues", []),
            "imbalance_ratio":       profile.get("imbalance_ratio"),
        },
        "features": {
            "redundant_pairs":  len(redundancy.get("redundant_pairs", [])),
            "drop_suggestions": redundancy.get("drop_suggestions", []),
            "skewed_features":  [k for k, v in distributions.items()
                                 if abs(v.get("skewness", 0)) > 1.5][:5],
            "type_warnings":    type_inference.get("warnings", []),
            "top_correlations": list(profile.get("top_correlations", {}).keys())[:5],
            "outlier_drivers":  (outlier_rca or {}).get("major_drivers", []),
            "outlier_pct":      (outlier_rca or {}).get("pct_outliers", 0),
        },
        "next_steps": {
            "phase1_count": len((fe_roadmap or {}).get("phase1", [])),
            "phase2_count": len((fe_roadmap or {}).get("phase2", [])),
            "phase3_count": len((fe_roadmap or {}).get("phase3", [])),
            "phase1_actions": [(s["action"]) for s in (fe_roadmap or {}).get("phase1", [])][:3],
            "phase2_actions": [(s["action"]) for s in (fe_roadmap or {}).get("phase2", [])][:3],
        },
        "missing": {
            "pattern":        missing_pattern.get("pattern", "NONE"),
            "recommendation": missing_pattern.get("recommendation", ""),
            "total_pct":      missing_pattern.get("total_missing_pct", 0),
        },
    }

    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)

            prompt = f"""You are a senior ML engineer writing a professional ML dataset audit report for a client.

Write 6 sections as flowing professional prose — NOT bullet points, NOT headers within sections.
Each section targets a DIFFERENT audience. Be specific — reference actual numbers and column names.
Write 3-5 sentences per section. Sound like a consultant, not a chatbot.

Return ONLY a valid JSON object with exactly these 6 string keys:
{{
  "executive_summary": "For non-technical managers. State the grade, headline risk, and one-line recommendation.",
  "dataset_quality_analysis": "For ML engineers. Cover missing data pattern, duplicates, outlier rate, integrity issues with exact numbers.",
  "risk_assessment": "For compliance/legal. Cover leakage risk, drift risk, time-series risk, and consequences of ignoring them.",
  "feature_analysis": "For data scientists. Cover redundancy, skewness, multicollinearity (VIF), top predictors, and what to transform.",
  "recommended_next_steps": "For project managers. Exact prioritised actions with effort estimates. Reference the 3-phase roadmap.",
  "model_recommendations": "For the engineering team. Which algorithm, why, what hyperparameters to tune first, what to avoid."
}}

ANALYSIS DATA:
{json.dumps(context, indent=2)}

Rules:
- Reference exact numbers (rows, columns, percentages, correlation values)
- Mention specific column names where relevant
- No markdown formatting inside strings
- No bullet points or numbered lists
- Each section must be a single paragraph of flowing prose
- Sound authoritative and expert, not generic"""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.35,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            parsed = json.loads(raw)
            required = ["executive_summary", "dataset_quality_analysis", "risk_assessment",
                        "feature_analysis", "recommended_next_steps", "model_recommendations"]
            if all(k in parsed for k in required):
                return {k: parsed[k] for k in required}
        except Exception:
            pass

    return _rule_based_report(context)


def _rule_based_report(ctx: dict) -> dict[str, str]:
    ds  = ctx["dataset"]
    q   = ctx["quality"]
    r   = ctx["risks"]
    f   = ctx["features"]
    ns  = ctx["next_steps"]
    ms  = ctx["missing"]

    leakage_note = ("No data leakage was detected, which is a positive signal for model validity. "
                     if not r["leakage_candidates"] else
                     ("The most critical finding is " + str(len(r["leakage_candidates"])) +
                      " potential leakage feature(s) which must be investigated immediately before any model training begins. "))
    rec_note = ("proceeding with the feature engineering roadmap outlined below before training"
                if q["score"] >= 60 else
                "resolving the critical data quality issues identified in this report before attempting any model training")
    exec_summary = (
        f"This dataset comprising {ds['rows']:,} rows and {ds['columns']} columns "
        f"targeting '{ds['target']}' for a {ds['task']} task has received a Data Quality Grade of "
        f"{q['grade']} ({q['score']}/100). "
        f"The health score of {q['health']}/100 reflects "
        f"{'strong data foundations with addressable issues' if q['health'] >= 70 else 'material data quality problems requiring resolution before training'}. "
        + leakage_note
        + f"We recommend {rec_note}."
    )

    quality_analysis = (
        f"The dataset contains {ds['missing_pct']}% missing values following a {ms['pattern']} pattern — "
        f"{ms['recommendation']} "
        f"There are {ds['duplicates']} duplicate rows which represent wasted compute and potential bias if not removed. "
        f"{'Outlier analysis using Mahalanobis distance detected ' + str(ctx.get('features',{}).get('outlier_pct',0)) + '% multivariate outliers, driven primarily by ' + str(f.get('outlier_drivers',['unknown'])[:2]) + '.' if f.get('outlier_pct', 0) > 0 else 'No significant multivariate outliers were detected.'} "
        f"The {ds['numeric']} numeric and {ds['categorical']} categorical features present "
        f"{'significant redundancy with ' + str(f['redundant_pairs']) + ' highly correlated pairs identified' if f['redundant_pairs'] > 0 else 'acceptable feature diversity with no major redundancy concerns'}."
    )

    risk_assessment = (
        f"{'CRITICAL RISK: ' + str(len(r['leakage_candidates'])) + ' feature(s) show correlation above 0.95 with the target, indicating near-certain data leakage: ' + str(r['leakage_candidates']) + '. Deploying a model trained on this data would produce fraudulently inflated metrics that collapse in production. ' if r['leakage_candidates'] else 'No data leakage was detected — feature correlations are within safe bounds. '}"
        f"{'Temporal drift analysis reveals ' + str(r['drift_pct']) + '% of features shift significantly between the first and second halves of this dataset, presenting a ' + r['drift_risk'] + ' deployment risk as production data will differ from training data. ' if r['drift_risk'] != 'UNKNOWN' else ''}"
        f"{'This is a time-series dataset with ' + r['ts_frequency'] + ' frequency. Using a random train/test split would constitute a methodological error by allowing future data to inform past predictions. ' if r['timeseries'] else ''}"
        f"{'Sample size adequacy check raised the following concern: ' + r['sample_issues'][0] if r['sample_issues'] else 'Sample size is adequate for the feature count.'}"
    )

    feature_analysis = (
        f"Feature redundancy analysis identified {f['redundant_pairs']} highly correlated pairs — "
        f"{'recommended drops include: ' + str(f['drop_suggestions'][:3]) + ', which carry near-duplicate information. ' if f['drop_suggestions'] else 'no critical redundancy requiring immediate action. '}"
        f"{'The following features show high skewness requiring log transformation before training with linear models: ' + str(f['skewed_features'][:3]) + '. ' if f['skewed_features'] else 'Feature distributions are generally well-behaved. '}"
        f"{'Smart column type inference flagged: ' + '; '.join(f['type_warnings'][:2]) + '. ' if f['type_warnings'] else 'No column type issues detected. '}"
        f"The top predictors by correlation with '{ds['target']}' are: {f['top_correlations'][:4]}, which should be prioritised in any feature selection or engineering work."
    )

    next_steps = (
        f"The feature engineering roadmap identifies {ns['phase1_count'] + ns['phase2_count'] + ns['phase3_count']} total improvements across three phases. "
        f"Phase 1 (Quick Wins, ~30 minutes): {', '.join(ns['phase1_actions'][:3]) if ns['phase1_actions'] else 'clean and encode basics'}. "
        f"Phase 2 (Moderate effort, 2-4 hours): {', '.join(ns['phase2_actions'][:2]) if ns['phase2_actions'] else 'engineer interactions and handle categoricals'}. "
        f"Phase 3 (Advanced, 1+ days): polynomial features, lag features, and external data enrichment. "
        f"{'FIRST ACTION: Remove the identified leakage feature(s) ' + str(r['leakage_candidates']) + ' before any other work. ' if r['leakage_candidates'] else ''}"
        f"Download the automated cleaned dataset from the Export section as your starting point."
    )

    model_rec = (
        f"Based on this {ds['task']} task with {ds['rows']:,} rows and {ds['numeric']} numeric features: "
        f"{'XGBoost or LightGBM are the recommended starting point — they handle mixed feature types natively, are robust to the outliers and skewness identified in this dataset, and require minimal preprocessing. ' if ds['numeric'] > 5 else 'Ridge Regression is appropriate for this compact numeric feature set and will generalize reliably with proper L2 regularisation. '}"
        f"{'For time-series data, use TimeSeriesSplit cross-validation with at least 5 folds — never random splitting. Consider adding lag features from the roadmap before training. ' if r['timeseries'] else 'Use stratified k-fold cross-validation (5 folds) for reliable performance estimates. '}"
        f"Tune n_estimators (100-500) and max_depth (3-7) first as these have the highest impact, and only proceed to learning_rate tuning after establishing a good tree structure. "
        + "Always compare against a naive baseline (mean predictor for regression, "
          "majority class for classification) to ensure the model delivers genuine value."
    )

    return {
        "executive_summary":         exec_summary,
        "dataset_quality_analysis":  quality_analysis,
        "risk_assessment":           risk_assessment,
        "feature_analysis":          feature_analysis,
        "recommended_next_steps":    next_steps,
        "model_recommendations":     model_rec,
    }


# ─────────────────────────────────────────────────────────────────
# PDF GENERATOR — Professional Audit Document
# ─────────────────────────────────────────────────────────────────

def generate_pdf(
    report_sections:  dict[str, str],
    profile:          dict,
    health:           dict,
    scorecard:        dict,
    leakage:          dict,
    redundancy:       dict,
    missing_pattern:  dict,
    sample_check:     dict,
    type_inference:   dict,
    target_column:    str,
    task_type:        str,
    fix_actions:      list[str],
    fe_roadmap:       dict | None = None,
    leakage_prob:     dict | None = None,
    drift_sim:        dict | None = None,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable, PageBreak,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2.2*cm, rightMargin=2.2*cm,
            topMargin=2*cm,    bottomMargin=2*cm,
        )

        NAVY  = colors.HexColor("#0f172a")
        BLUE  = colors.HexColor("#2563eb")
        LGRAY = colors.HexColor("#f1f5f9")
        MGRAY = colors.HexColor("#94a3b8")
        GREEN = colors.HexColor("#16a34a")
        AMBER = colors.HexColor("#d97706")
        RED   = colors.HexColor("#dc2626")

        grade_color = {
            "A": GREEN,
            "B": colors.HexColor("#22c55e"),
            "C": AMBER,
            "D": colors.HexColor("#ea580c"),
            "F": RED,
        }

        s_title  = ParagraphStyle("T",  fontSize=24, fontName="Helvetica-Bold",
                                   textColor=NAVY, spaceAfter=4)
        s_sub    = ParagraphStyle("S",  fontSize=11, fontName="Helvetica",
                                   textColor=MGRAY, spaceAfter=4)
        s_h1     = ParagraphStyle("H1", fontSize=15, fontName="Helvetica-Bold",
                                   textColor=NAVY, spaceBefore=16, spaceAfter=5)
        s_h2     = ParagraphStyle("H2", fontSize=11, fontName="Helvetica-Bold",
                                   textColor=BLUE, spaceBefore=10, spaceAfter=4)
        s_body   = ParagraphStyle("BD", fontSize=9.5, fontName="Helvetica",
                                   leading=15, spaceAfter=6,
                                   textColor=colors.HexColor("#334155"))
        s_code   = ParagraphStyle("CD", fontSize=8, fontName="Courier",
                                   leading=12, spaceAfter=4,
                                   textColor=colors.HexColor("#1e40af"),
                                   backColor=colors.HexColor("#f0f4ff"))
        s_cap    = ParagraphStyle("CA", fontSize=7.5, fontName="Helvetica-Oblique",
                                   textColor=MGRAY, spaceAfter=2)

        def tbl(data, widths, hdr_bg=NAVY):
            t = Table(data, colWidths=widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), hdr_bg),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGRAY]),
                ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
                ("PADDING",        (0,0), (-1,-1), 6),
                ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
            ]))
            return t

        SECTION_TITLES = {
            "executive_summary":        "Executive Summary",
            "dataset_quality_analysis": "Dataset Quality Analysis",
            "risk_assessment":          "Risk Assessment",
            "feature_analysis":         "Feature Analysis",
            "recommended_next_steps":   "Recommended Next Steps",
            "model_recommendations":    "Model Recommendations",
        }

        SECTION_AUDIENCE = {
            "executive_summary":        "For: Management",
            "dataset_quality_analysis": "For: ML Engineers",
            "risk_assessment":          "For: Compliance",
            "feature_analysis":         "For: Data Scientists",
            "recommended_next_steps":   "For: Project Managers",
            "model_recommendations":    "For: Engineering Team",
        }

        story = []

        # ── COVER PAGE ────────────────────────────────────────────
        story.append(Spacer(1, 1.2*cm))
        story.append(Paragraph("AutoML Debugger", s_sub))
        story.append(Paragraph("ML Dataset Audit Report", s_title))
        story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=14))

        grade  = scorecard.get("overall_grade", "?")
        gc     = grade_color.get(grade, NAVY)
        now_str = datetime.now().strftime("%B %d, %Y  %H:%M")

        meta = [
            ["Target Column",  target_column,
             "Task Type",      task_type.capitalize()],
            ["Dataset Size",   f"{profile.get('rows',0):,} rows x {profile.get('columns',0)} cols",
             "Generated",      now_str],
            ["Overall Grade",  grade,
             "Score",          f"{scorecard.get('overall_score',0)}/100"],
            ["Health Score",   f"{health.get('total',0)}/100",
             "Verdict",        scorecard.get("overall_verdict","")],
        ]
        mt = Table(meta, colWidths=[4*cm, 5*cm, 3.5*cm, 4.5*cm])
        mt.setStyle(TableStyle([
            ("FONTSIZE",       (0,0), (-1,-1), 8.5),
            ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",       (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR",      (0,0), (0,-1), NAVY),
            ("TEXTCOLOR",      (2,0), (2,-1), NAVY),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, LGRAY]),
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("PADDING",        (0,0), (-1,-1), 7),
        ]))
        story.append(mt)
        story.append(Spacer(1, 0.5*cm))

        gc_hex = gc.hexval()[2:]
        story.append(Paragraph(
            f'<font size="13" color="#{gc_hex}"><b>Data Quality Grade: {grade}</b></font>',
            s_body,
        ))
        verdict_clean = scorecard.get("overall_verdict","").replace("\u2014", "-")
        story.append(Paragraph(
            f'<font size="11" color="#{gc_hex}">{verdict_clean}</font>',
            s_body,
        ))
        story.append(Paragraph(scorecard.get("benchmark",""), s_cap))
        story.append(Spacer(1, 0.6*cm))
        story.append(PageBreak())

        # ── SCORECARD ─────────────────────────────────────────────
        story.append(Paragraph("Data Quality Scorecard", s_h1))
        sc_data = [["Section", "Score", "Grade", "Impact", "Details"]]
        for sec in scorecard.get("sections", []):
            sc_data.append([
                sec["name"],
                f"{sec['score']}/100",
                sec["grade"],
                sec.get("impact",""),
                sec.get("details","")[:55],
            ])
        story.append(tbl(sc_data, [4.5*cm, 2*cm, 1.5*cm, 2*cm, 7*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Health breakdown
        story.append(Paragraph("Health Score Breakdown", s_h2))
        hd = [["Dimension", "Score", "Max", "Notes"]]
        for dim, dv in health.get("dimensions", {}).items():
            hd.append([dim, str(dv["score"]), str(dv["max"]), dv.get("reason","")[:60]])
        hd.append(["TOTAL", str(health.get("total",0)), "100", health.get("verdict","")])
        story.append(tbl(hd, [5*cm, 2*cm, 2*cm, 8*cm]))
        story.append(PageBreak())

        # ── GROQ REPORT SECTIONS ──────────────────────────────────
        for key_name, title in SECTION_TITLES.items():
            text = report_sections.get(key_name, "")
            if not text:
                continue
            story.append(Paragraph(title, s_h1))
            story.append(Paragraph(SECTION_AUDIENCE[key_name], s_cap))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#e2e8f0"), spaceAfter=8))
            story.append(Paragraph(text, s_body))
            story.append(Spacer(1, 0.2*cm))

        story.append(PageBreak())

        # ── DATA TABLES ───────────────────────────────────────────
        story.append(Paragraph("Dataset Profile", s_h1))
        prof = [
            ["Metric", "Value", "Metric", "Value"],
            ["Rows",             f"{profile.get('rows',0):,}",
             "Numeric Features",  str(profile.get('numeric_features',0))],
            ["Columns",          str(profile.get('columns',0)),
             "Categorical",       str(profile.get('categorical_features',0))],
            ["Missing Values",   f"{profile.get('missing_pct',0)}%",
             "Duplicates",        str(profile.get('duplicate_rows',0))],
            ["Constant Features", str(len(profile.get('constant_features',[]))),
             "High-Card Cols",    str(len(profile.get('high_cardinality_cols',[])))],
            ["Leakage Candidates", str(len(leakage.get('leakage_candidates',[]))),
             "Redundant Pairs",   str(len(redundancy.get('redundant_pairs',[])))],
        ]
        story.append(tbl(prof, [4.5*cm, 4*cm, 4.5*cm, 4*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Missing pattern
        story.append(Paragraph("Missing Value Pattern", s_h2))
        mp_d = [
            ["Pattern", "Total Missing %", "Recommendation"],
            [missing_pattern.get("pattern","NONE"),
             f"{missing_pattern.get('total_missing_pct',0)}%",
             missing_pattern.get("recommendation","")[:80]],
        ]
        story.append(tbl(mp_d, [3*cm, 4*cm, 10*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Redundant pairs
        red_pairs = redundancy.get("redundant_pairs", [])
        if red_pairs:
            story.append(Paragraph("Top Redundant Feature Pairs", s_h2))
            rp = [["Feature 1","Feature 2","Correlation","Severity"]]
            for p in red_pairs[:8]:
                rp.append([p["feature_1"], p["feature_2"],
                            str(p["correlation"]), p["severity"]])
            story.append(tbl(rp, [4.5*cm, 4.5*cm, 3*cm, 5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Leakage probability
        if leakage_prob and leakage_prob.get("features"):
            story.append(Paragraph("Leakage Probability Scores (Top 10)", s_h2))
            lp_d = [["Feature","MI Score","Correlation","Leakage Prob","Risk"]]
            for col, v in list(leakage_prob["features"].items())[:10]:
                lp_d.append([col, str(v["mi_score"]), str(v["correlation"]),
                              str(v["leakage_prob"]), v["risk_level"]])
            story.append(tbl(lp_d, [4*cm, 2.5*cm, 3*cm, 3*cm, 4.5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Drift simulation
        if drift_sim and drift_sim.get("available"):
            story.append(Paragraph("Data Drift Simulation (First vs Second Half)", s_h2))
            dr_top = drift_sim.get("drift_results", [])[:8]
            if dr_top:
                dr_d = [["Feature","KS Stat","Drifted","Severity","Mean Shift %"]]
                for r in dr_top:
                    dr_d.append([r["feature"], str(r["ks_stat"]),
                                  "YES" if r["drifted"] else "NO",
                                  r["severity"], f"{r['mean_shift']}%"])
                story.append(tbl(dr_d, [4*cm, 2.5*cm, 2*cm, 3*cm, 3*cm]))
                story.append(Spacer(1, 0.4*cm))

        # Feature Engineering Roadmap
        if fe_roadmap:
            story.append(Paragraph("Feature Engineering Roadmap", s_h2))
            for phase_key, phase_label in [
                ("phase1","Phase 1 - Quick Wins (~30 min)"),
                ("phase2","Phase 2 - Moderate Effort (2-4 hours)"),
                ("phase3","Phase 3 - Advanced (1+ days)"),
            ]:
                items = fe_roadmap.get(phase_key, [])
                if not items:
                    continue
                story.append(Paragraph(phase_label, s_cap))
                ph_d = [["Action","Columns","Expected Impact","Code"]]
                for item in items:
                    cols_str = str(item.get("columns",""))[:30]
                    code_str = item.get("code","").split("\n")[0][:40]
                    ph_d.append([
                        item.get("action","")[:40],
                        cols_str,
                        item.get("expected_impact","")[:40],
                        code_str,
                    ])
                story.append(tbl(ph_d, [4.5*cm, 3.5*cm, 4*cm, 5*cm]))
                story.append(Spacer(1, 0.2*cm))
            story.append(Spacer(1, 0.2*cm))

        # Sample size checks
        checks = sample_check.get("checks", [])
        if checks:
            story.append(Paragraph("Sample Size Adequacy", s_h2))
            ch = [["Rule","Needed","Have","Status"]]
            for c in checks:
                ch.append([c["rule"], str(c["needed"]), str(c["have"]),
                            "Pass" if c["pass"] else "FAIL"])
            story.append(tbl(ch, [7*cm, 3*cm, 3*cm, 4*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Automated fixes
        if fix_actions:
            story.append(Paragraph("Automated Fixes Applied to Cleaned Dataset", s_h2))
            fx = [["#","Action"]]
            for i, a in enumerate(fix_actions, 1):
                fx.append([str(i), a])
            story.append(tbl(fx, [1.5*cm, 15.5*cm]))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return (
            f"AUTOML DEBUGGER AUDIT REPORT\n"
            f"Generated: {datetime.now()}\n"
            f"Grade: {scorecard.get('overall_grade','?')}\n"
            f"Score: {scorecard.get('overall_score',0)}/100\n"
        ).encode()
