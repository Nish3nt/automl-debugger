"""
AutoML Debugger — Professional Report Generator  (Milestone 1 v4.0)
====================================================================
Groq writes each section as real paragraphs — not bullet points.
Sections:
  1. Executive Summary        (for managers)
  2. Dataset Quality Analysis (for engineers)
  3. Risk Assessment          (for compliance)
  4. Feature Analysis         (for data scientists)
  5. Recommended Next Steps   (for project planning)
  6. Model Recommendations    (which model family to use and why)

PDF is structured like a professional audit document.
"""

from __future__ import annotations

import io
import os
import json
from datetime import datetime
from typing import Any


# ─────────────────────────────────────────────────────────────────
# GROQ REPORT WRITER
# ─────────────────────────────────────────────────────────────────

def generate_groq_report(
    profile:         dict,
    health:          dict,
    scorecard:       dict,
    leakage:         dict,
    redundancy:      dict,
    missing_pattern: dict,
    distributions:   dict,
    type_inference:  dict,
    sample_check:    dict,
    ts_info:         dict,
    target_column:   str,
    task_type:       str,
    api_key:         str | None = None,
) -> dict[str, str]:
    """
    Returns dict with section_name -> paragraph text.
    Falls back to rule-based text if no API key.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    context = {
        "dataset": {
            "rows":       profile.get("rows", 0),
            "columns":    profile.get("columns", 0),
            "numeric":    profile.get("numeric_features", 0),
            "categorical":profile.get("categorical_features", 0),
            "missing_pct":profile.get("missing_pct", 0),
            "duplicates": profile.get("duplicate_rows", 0),
            "target":     target_column,
            "task":       task_type,
        },
        "health": {
            "total":   health.get("total", 0),
            "grade":   health.get("grade", "?"),
            "verdict": health.get("verdict", ""),
            "dims":    {k: v["score"] for k, v in health.get("dimensions", {}).items()},
        },
        "scorecard": {
            "overall":  scorecard.get("overall_score", 0),
            "grade":    scorecard.get("overall_grade", "?"),
            "verdict":  scorecard.get("overall_verdict", ""),
            "sections": [{
                "name":  s["name"],
                "score": s["score"],
                "grade": s["grade"],
            } for s in scorecard.get("sections", [])],
        },
        "leakage": {
            "candidates": leakage.get("leakage_candidates", []),
            "n_high_corr": len([v for v in leakage.get("high_correlation_features", {}).values() if v > 0.85]),
        },
        "redundancy": {
            "n_pairs":       len(redundancy.get("redundant_pairs", [])),
            "drop_suggestions": redundancy.get("drop_suggestions", []),
            "top_vif":       sorted(redundancy.get("vif_scores", {}).items(), key=lambda x: x[1], reverse=True)[:3],
        },
        "missing": {
            "pattern":        missing_pattern.get("pattern", "NONE"),
            "recommendation": missing_pattern.get("recommendation", ""),
            "total_pct":      missing_pattern.get("total_missing_pct", 0),
        },
        "sample": {
            "adequate": sample_check.get("adequate", True),
            "issues":   sample_check.get("issues", []),
            "projected_improvement": sample_check.get("projected_improvement", ""),
        },
        "timeseries": {
            "detected":  ts_info.get("is_timeseries", False),
            "frequency": ts_info.get("frequency_guess", ""),
            "column":    ts_info.get("datetime_column", ""),
        },
        "distributions": {
            "highly_skewed": [
                col for col, d in distributions.items() if abs(d.get("skewness", 0)) > 1.5
            ][:5],
            "non_normal": [
                col for col, d in distributions.items()
                if not d.get("is_normal", True) and d.get("p_normal") is not None
            ][:5],
        },
    }

    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)

            prompt = f"""You are a senior ML engineer writing a professional dataset audit report.

Based on the analysis data below, write 6 sections of a professional report.
Each section should be 3-5 sentences written as flowing professional prose (NOT bullet points).
Be specific — reference actual numbers from the data.
Write as if you're a consultant presenting findings to a client.

Return ONLY a valid JSON object with exactly these 6 keys, each containing a string:
{{
  "executive_summary": "...",
  "dataset_quality_analysis": "...",
  "risk_assessment": "...",
  "feature_analysis": "...",
  "recommended_next_steps": "...",
  "model_recommendations": "..."
}}

ANALYSIS DATA:
{json.dumps(context, indent=2)}

Section guidelines:
- executive_summary: High-level verdict for a non-technical manager. Mention the grade, key strength, and biggest concern.
- dataset_quality_analysis: Technical details for an ML engineer. Cover missing data pattern, duplicates, outliers, and data integrity.
- risk_assessment: Focus on leakage risks, time-series risks, class imbalance risks. Be direct about consequences.
- feature_analysis: Discuss redundancy, VIF scores, skewed distributions, and which features need transformation.
- recommended_next_steps: Exact prioritized actions. Be concrete — "Drop column X", "Apply log transform to Y", etc.
- model_recommendations: Based on the data profile, which model family to use and why. Mention specific algorithms.

Rules: Professional tone. No markdown formatting inside strings. No bullet points. Pure flowing prose. Reference actual numbers."""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.4,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            parsed = json.loads(raw)
            required = ["executive_summary", "dataset_quality_analysis", "risk_assessment",
                        "feature_analysis", "recommended_next_steps", "model_recommendations"]
            if all(k in parsed for k in required):
                return parsed
        except Exception:
            pass

    return _rule_based_report(context)


def _rule_based_report(ctx: dict) -> dict[str, str]:
    ds  = ctx["dataset"]
    h   = ctx["health"]
    sc  = ctx["scorecard"]
    lk  = ctx["leakage"]
    rd  = ctx["redundancy"]
    ms  = ctx["missing"]
    sa  = ctx["sample"]
    ts  = ctx["timeseries"]
    di  = ctx["distributions"]

    exec_summary = (
        f"This dataset contains {ds['rows']:,} rows and {ds['columns']} columns targeting "
        f"'{ds['target']}' for a {ds['task']} task. "
        f"The overall data quality grade is {sc['grade']} ({sc['overall']}%) — {sc['verdict']}. "
        f"The health score of {h['total']}/100 reflects "
        f"{'strong data quality with minor issues to address' if h['total'] >= 75 else 'significant data quality issues that must be resolved before training'}. "
        + ("No data leakage was detected, which is a positive indicator." if not lk["candidates"] else ("Critical attention is required for " + str(len(lk["candidates"])) + " potential leakage feature(s)."))
    )

    quality_analysis = (
        f"The dataset has {ds['missing_pct']}% missing values following a {ms['pattern']} pattern — "
        f"{ms['recommendation']} "
        f"There are {ds['duplicates']} duplicate rows which should be removed before training. "
        f"{'Outlier analysis reveals skewed distributions in several features requiring transformation. ' if di['highly_skewed'] else 'Feature distributions are generally well-behaved. '}"
        f"The {ds['numeric']} numeric and {ds['categorical']} categorical features provide "
        f"{'a rich diverse feature set' if ds['numeric'] + ds['categorical'] > 10 else 'a compact feature set'}."
    )

    risk_assessment = (
        f"{'CRITICAL: ' + str(len(lk['candidates'])) + ' feature(s) show correlation >0.95 with the target, indicating high leakage risk: ' + str(lk['candidates']) + '. ' if lk['candidates'] else 'No data leakage risk detected — all feature correlations are within safe bounds. '}"
        f"{'This is a time-series dataset with ' + ts['frequency'] + ' frequency. Using a random train/test split would leak future information into training — chronological splitting is mandatory. ' if ts['detected'] else ''}"
        f"{'Sample size adequacy check found issues: ' + '; '.join(sa['issues']) if sa['issues'] else 'Sample size is adequate for the number of features. '}"
        f"Overall risk level is {'HIGH' if lk['candidates'] else ('MEDIUM' if sa['issues'] else 'LOW')}."
    )

    feature_analysis = (
        f"Feature redundancy analysis identified {rd['n_pairs']} highly correlated feature pairs. "
        f"{'Suggested drops: ' + str(rd['drop_suggestions'][:3]) + ' to reduce multicollinearity. ' if rd['drop_suggestions'] else 'No critical redundancy detected. '}"
        f"{'The following features show high skewness and would benefit from log transformation: ' + str(di['highly_skewed']) + '. ' if di['highly_skewed'] else 'Feature distributions are generally symmetric. '}"
        f"{'Features with high VIF scores indicate multicollinearity that may destabilize linear models. ' if rd.get('top_vif') else ''}"
        f"Correlation analysis with the target column reveals the most predictive features."
    )

    next_steps = (
        f"{'FIRST: Remove leakage features ' + str(lk['candidates']) + ' before any training. ' if lk['candidates'] else ''}"
        f"{'Drop redundant features: ' + str(rd['drop_suggestions'][:3]) + '. ' if rd['drop_suggestions'] else ''}"
        f"{'Apply log1p transform to skewed features: ' + str(di['highly_skewed'][:3]) + '. ' if di['highly_skewed'] else ''}"
        f"Remove {ds['duplicates']} duplicate rows using the cleaned dataset download. "
        f"{'Use chronological train/test split — do NOT use random splitting on this time-series data. ' if ts['detected'] else 'Use stratified random split for train/test separation. '}"
        f"Download the automated fixed dataset from the Export section as your starting point."
    )

    model_rec = (
        f"Based on this {ds['task']} task with {ds['rows']:,} rows and {ds['numeric']} numeric features: "
        f"{'XGBoost or LightGBM are strongly recommended — they handle the mixed numeric/categorical features natively, are robust to outliers, and do not require scaling. ' if ds['numeric'] > 5 else 'Ridge Regression is appropriate for this small feature set and will generalize well with proper regularization. '}"
        f"{'For time-series data, use TimeSeriesSplit cross-validation and consider lag features. ' if ts['detected'] else ''}"
        f"{'With significant class imbalance, use class_weight=\"balanced\" parameter. ' if ctx.get('imbalance_ratio', 1) > 3 else ''}"
        f"Start with default hyperparameters and tune only n_estimators and max_depth in a second iteration."
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
    report_sections: dict[str, str],
    profile:         dict,
    health:          dict,
    scorecard:       dict,
    leakage:         dict,
    redundancy:      dict,
    missing_pattern: dict,
    sample_check:    dict,
    type_inference:  dict,
    target_column:   str,
    task_type:       str,
    fix_actions:     list[str],
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

        # ── Styles ────────────────────────────────────────────────
        NAVY  = colors.HexColor("#1a2035")
        BLUE  = colors.HexColor("#2563eb")
        LGRAY = colors.HexColor("#f1f5f9")
        MGRAY = colors.HexColor("#94a3b8")
        GREEN = colors.HexColor("#16a34a")
        AMBER = colors.HexColor("#d97706")
        RED   = colors.HexColor("#dc2626")

        grade_color = {
            "A": GREEN, "B": colors.HexColor("#22c55e"),
            "C": AMBER,  "D": colors.HexColor("#ea580c"),
            "F": RED,
        }

        s_cover_title = ParagraphStyle("CT", fontSize=26, fontName="Helvetica-Bold",
                                        textColor=NAVY, spaceAfter=6)
        s_cover_sub   = ParagraphStyle("CS", fontSize=12, fontName="Helvetica",
                                        textColor=MGRAY, spaceAfter=4)
        s_h1          = ParagraphStyle("H1", fontSize=16, fontName="Helvetica-Bold",
                                        textColor=NAVY, spaceBefore=16, spaceAfter=6)
        s_h2          = ParagraphStyle("H2", fontSize=12, fontName="Helvetica-Bold",
                                        textColor=BLUE, spaceBefore=10, spaceAfter=4)
        s_body        = ParagraphStyle("BD", fontSize=9.5, fontName="Helvetica",
                                        leading=15, spaceAfter=6, textColor=colors.HexColor("#334155"))
        s_caption     = ParagraphStyle("CA", fontSize=7.5, fontName="Helvetica-Oblique",
                                        textColor=MGRAY, spaceAfter=2)
        s_label       = ParagraphStyle("LB", fontSize=8, fontName="Helvetica-Bold",
                                        textColor=MGRAY, spaceAfter=2)

        def make_table(data, col_widths, header_bg=NAVY):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), header_bg),
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

        story = []

        # ── COVER PAGE ────────────────────────────────────────────
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("AutoML Debugger", s_cover_sub))
        story.append(Paragraph("ML Dataset Audit Report", s_cover_title))
        story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=14))

        grade    = scorecard.get("overall_grade", "?")
        gc       = grade_color.get(grade, NAVY)
        gen_time = datetime.now().strftime("%B %d, %Y  %H:%M")

        meta_data = [
            ["Target Column",  target_column,     "Task Type",   task_type.capitalize()],
            ["Dataset Size",   f"{profile.get('rows',0):,} rows × {profile.get('columns',0)} cols",
             "Generated",      gen_time],
            ["Overall Grade",  grade,              "Overall Score", f"{scorecard.get('overall_score',0)}/100"],
            ["Verdict",        scorecard.get("overall_verdict",""), "Health Score", f"{health.get('total',0)}/100"],
        ]
        mt = Table(meta_data, colWidths=[4*cm, 5.5*cm, 4*cm, 3.5*cm])
        mt.setStyle(TableStyle([
            ("FONTSIZE",  (0,0), (-1,-1), 8.5),
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,0), (0,-1), NAVY),
            ("TEXTCOLOR", (2,0), (2,-1), NAVY),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, LGRAY]),
            ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("PADDING",   (0,0), (-1,-1), 7),
        ]))
        story.append(mt)
        story.append(Spacer(1, 0.5*cm))

        # Grade badge
        story.append(Paragraph(
            f'<font size="14" color="#{gc.hexval()[2:]}"><b>Data Quality Grade: {grade} — {scorecard.get("overall_verdict","")}</b></font>',
            s_body,
        ))
        story.append(Paragraph(scorecard.get("benchmark", ""), s_caption))
        story.append(PageBreak())

        # ── SCORECARD TABLE ───────────────────────────────────────
        story.append(Paragraph("Data Quality Scorecard", s_h1))
        sc_data = [["Section", "Score", "Grade", "Impact", "Details"]]
        for sec in scorecard.get("sections", []):
            g    = sec["grade"]
            gcol = grade_color.get(g, NAVY)
            sc_data.append([
                sec["name"],
                f"{sec['score']}/100",
                sec["grade"],
                sec.get("impact", ""),
                sec.get("details", "")[:60],
            ])
        story.append(make_table(sc_data, [4.5*cm, 2*cm, 1.5*cm, 2*cm, 7*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Health score breakdown
        story.append(Paragraph("Health Score Breakdown", s_h2))
        hd_data = [["Dimension", "Score", "Max", "Notes"]]
        for dim, dv in health.get("dimensions", {}).items():
            hd_data.append([dim, str(dv["score"]), str(dv["max"]), dv.get("reason", "")])
        hd_data.append(["TOTAL", str(health.get("total",0)), "100", health.get("verdict","")])
        story.append(make_table(hd_data, [5*cm, 2*cm, 2*cm, 8*cm]))
        story.append(PageBreak())

        # ── GROQ REPORT SECTIONS ──────────────────────────────────
        for key_name, title in SECTION_TITLES.items():
            text = report_sections.get(key_name, "")
            if not text:
                continue
            story.append(Paragraph(title, s_h1))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
            story.append(Paragraph(text, s_body))
            story.append(Spacer(1, 0.3*cm))

        story.append(PageBreak())

        # ── DATA PROFILE TABLE ────────────────────────────────────
        story.append(Paragraph("Dataset Profile", s_h1))
        prof_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Rows",    f"{profile.get('rows',0):,}",
             "Numeric Features", str(profile.get('numeric_features',0))],
            ["Total Columns", str(profile.get('columns',0)),
             "Categorical Features", str(profile.get('categorical_features',0))],
            ["Missing Values", f"{profile.get('missing_pct',0)}%",
             "Duplicate Rows", str(profile.get('duplicate_rows',0))],
            ["Constant Features", str(len(profile.get('constant_features',[]))),
             "High-Cardinality Cols", str(len(profile.get('high_cardinality_cols',[])))],
            ["Outlier Columns", str(len(profile.get('outlier_counts',{}))),
             "Leakage Candidates", str(len(leakage.get('leakage_candidates',[])))],
        ]
        story.append(make_table(prof_data, [4.5*cm, 4*cm, 4.5*cm, 4*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Missing pattern
        story.append(Paragraph("Missing Value Pattern", s_h2))
        mp = missing_pattern
        mp_data = [
            ["Pattern", "Total Missing %", "Recommendation"],
            [mp.get("pattern","NONE"), f"{mp.get('total_missing_pct',0)}%", mp.get("recommendation","")[:80]],
        ]
        story.append(make_table(mp_data, [3*cm, 4*cm, 10*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Redundant pairs
        red_pairs = redundancy.get("redundant_pairs", [])
        if red_pairs:
            story.append(Paragraph("Feature Redundancy", s_h2))
            rp_data = [["Feature 1", "Feature 2", "Correlation", "Severity"]]
            for p in red_pairs[:10]:
                rp_data.append([
                    p["feature_1"], p["feature_2"],
                    str(p["correlation"]), p["severity"],
                ])
            story.append(make_table(rp_data, [4.5*cm, 4.5*cm, 3*cm, 5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Sample size checks
        checks = sample_check.get("checks", [])
        if checks:
            story.append(Paragraph("Sample Size Adequacy", s_h2))
            ch_data = [["Rule", "Needed", "Have", "Status"]]
            for c in checks:
                ch_data.append([
                    c["rule"],
                    str(c["needed"]),
                    str(c["have"]),
                    "✅ Pass" if c["pass"] else "❌ Fail",
                ])
            story.append(make_table(ch_data, [7*cm, 3*cm, 3*cm, 4*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Automated fixes applied
        if fix_actions:
            story.append(Paragraph("Automated Fixes Applied to Cleaned Dataset", s_h2))
            fx_data = [["#", "Action"]]
            for i, a in enumerate(fix_actions, 1):
                fx_data.append([str(i), a])
            story.append(make_table(fx_data, [1.5*cm, 15.5*cm]))

        # ── FOOTER ────────────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY))
        story.append(Paragraph(
            f"AutoML Debugger v4.0 · Milestone 1 · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            f"Powered by Groq LLaMA 3.3 70B · Built with Streamlit",
            s_caption,
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return f"AUTOML DEBUGGER REPORT\nGenerated: {datetime.now()}\nGrade: {scorecard.get('overall_grade','?')}\n".encode()
