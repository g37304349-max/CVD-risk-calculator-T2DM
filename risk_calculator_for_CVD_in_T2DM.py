import math

import numpy as np
import streamlit as st


# ============================================================
# 1. Protein risk score weights
# Protein inputs are assumed to be Olink NPX values on the same
# scale as the UK Biobank proteomics data used for model development.
# The calculator does not re-normalize protein inputs.
# ============================================================

protein_weights = {
    "ACAA1": -0.4026,
    "ALPP": -0.0284,
    "HMOX1": 0.4618,
    "IL18RAP": -0.1533,
    "ITGB1BP1": 0.8919,
    "MICB_MICA": 0.1887,
    "MUC16": -0.0831,
    "NTproBNP": 0.4211,
    "REN": 0.3902,
    "TIMP3": 0.1159,
    "VIM": 0.1217,
    "XPNPEP2": 0.0954,
}

protein_display_names = {
    "ACAA1": "ACAA1",
    "ALPP": "ALPP",
    "HMOX1": "HMOX1",
    "IL18RAP": "IL18RAP",
    "ITGB1BP1": "ITGB1BP1",
    "MICB_MICA": "MICB/MICA",
    "MUC16": "MUC16",
    "NTproBNP": "NT-proBNP",
    "REN": "REN",
    "TIMP3": "TIMP3",
    "VIM": "VIM",
    "XPNPEP2": "XPNPEP2",
}


# ============================================================
# 2. Training-set standardization parameters for clinical variables
# From standardization_parameters_from_training_set.csv
# ============================================================

standardization_params = {
    "age": {"mean": 59.793496, "sd": 7.108997},
    "SBP": {"mean": 144.107640, "sd": 17.731316},
    "diab": {"mean": 52.374729, "sd": 9.193202},
    "HDL.c": {"mean": 1.138337, "sd": 0.265358},
    "cholesterol": {"mean": 4.388545, "sd": 0.962797},
    "HbA1c": {"mean": 52.149975, "sd": 13.383983},
    "eGFR": {"mean": 87.272670, "sd": 7.394785},
}


# ============================================================
# 3. Model coefficients
# ============================================================

prs_model_intercept = -1.6732
prs_model_beta = 1.0082

clinical_prs_coef = {
    "Intercept": -2.1313,
    "PRS": 0.8958,
    "age": 0.3627,
    "sex": 0.4781,
    "HbA1c": 0.1742,
    "SBP": -0.1629,
    "diab": -0.1589,
    "eGFR": -0.1826,
    "smoking_status": 0.1200,
    "HDL.c": -0.0321,
    "cholesterol": 0.0057,
}


clinical_fields = {
    "age": "Age, years",
    "sex": "Sex",
    "HbA1c": "HbA1c, mmol/mol",
    "SBP": "Systolic blood pressure, mmHg",
    "diab": "Age at diabetes diagnosis, years",
    "eGFR": "eGFR, mL/min/1.73m2",
    "smoking_status": "Smoking status",
    "HDL.c": "HDL-c, mmol/L",
    "cholesterol": "Total cholesterol, mmol/L",
}


# ============================================================
# 4. Utility functions
# ============================================================

def logistic(lp):
    return 1 / (1 + np.exp(-lp))


def standardize(value, variable):
    mean = standardization_params[variable]["mean"]
    sd = standardization_params[variable]["sd"]
    return (value - mean) / sd


def calculate_prs(protein_values):
    return sum(protein_values[protein] * weight for protein, weight in protein_weights.items())


def calculate_prs_model_risk(prs):
    lp = prs_model_intercept + prs_model_beta * prs
    return logistic(lp)


def calculate_clinical_prs_model_risk(
    prs,
    age,
    sex,
    hba1c,
    sbp,
    diab,
    egfr,
    smoking_status,
    hdl_c,
    cholesterol,
):
    age_std = standardize(age, "age")
    hba1c_std = standardize(hba1c, "HbA1c")
    sbp_std = standardize(sbp, "SBP")
    diab_std = standardize(diab, "diab")
    egfr_std = standardize(egfr, "eGFR")
    hdl_std = standardize(hdl_c, "HDL.c")
    cholesterol_std = standardize(cholesterol, "cholesterol")

    lp = (
        clinical_prs_coef["Intercept"]
        + clinical_prs_coef["PRS"] * prs
        + clinical_prs_coef["age"] * age_std
        + clinical_prs_coef["sex"] * sex
        + clinical_prs_coef["HbA1c"] * hba1c_std
        + clinical_prs_coef["SBP"] * sbp_std
        + clinical_prs_coef["diab"] * diab_std
        + clinical_prs_coef["eGFR"] * egfr_std
        + clinical_prs_coef["smoking_status"] * smoking_status
        + clinical_prs_coef["HDL.c"] * hdl_std
        + clinical_prs_coef["cholesterol"] * cholesterol_std
    )

    return logistic(lp)


def parse_float_input(raw_value):
    value = raw_value.strip()
    if value == "":
        return None

    try:
        parsed = float(value)
    except ValueError:
        return math.nan

    if not math.isfinite(parsed):
        return math.nan
    return parsed


def missing_labels(values, labels):
    return [labels[key] for key, value in values.items() if value is None]


def invalid_labels(values, labels):
    return [labels[key] for key, value in values.items() if isinstance(value, float) and math.isnan(value)]


# ============================================================
# 5. Streamlit page
# ============================================================

st.set_page_config(page_title="CVD Risk Calculator", layout="centered")

st.title("10-year CVD Risk Calculator for Patients with T2DM")

st.markdown(
    """
This calculator estimates the predicted 10-year risk of cardiovascular disease in patients
with type 2 diabetes mellitus using two models:

- **PRS model:** requires all 12 Olink protein values.
- **Clinical-PRS model:** requires all 12 Olink protein values plus the optional clinical variables below.

The calculator does not impute missing values. A prediction is generated only when all variables
required by that model are provided.
"""
)

st.warning(
    "This calculator is for research use only and should not replace clinical judgment."
)

st.info(
    """
Protein inputs must be Olink normalized protein expression (NPX) values on the same scale as
the UK Biobank proteomics data used for model development. NPX is a relative expression measure
on a log2 scale after Olink/UKB normalization and quality control. Raw protein concentrations,
raw assay counts, or values from other platforms should not be entered directly. The calculator
does not convert raw protein data to NPX and does not further standardize protein inputs before
calculating PRS.
"""
)


st.header("Protein Inputs")
st.caption(
    "Required for the PRS model and the Clinical-PRS model. Leave blank if unavailable; no protein-based prediction will be generated."
)

protein_values = {}
protein_columns = st.columns(3)

for index, protein in enumerate(protein_weights):
    with protein_columns[index % 3]:
        protein_values[protein] = parse_float_input(
            st.text_input(
                label=f"{protein_display_names[protein]} NPX",
                value="",
                placeholder="Olink NPX value",
                key=f"protein_{protein}",
            )
        )


st.header("Clinical Inputs (optional)")
st.caption(
    "Required only for the Clinical-PRS model. Clinical variables are standardized internally using training-set parameters."
)

clinical_col_1, clinical_col_2 = st.columns(2)

with clinical_col_1:
    age = parse_float_input(st.text_input("Age, years", value="", placeholder="e.g., 60"))
    hba1c = parse_float_input(st.text_input("HbA1c, mmol/mol", value="", placeholder="e.g., 52"))
    sbp = parse_float_input(st.text_input("Systolic blood pressure, mmHg", value="", placeholder="e.g., 144"))
    diab = parse_float_input(st.text_input("Age at diabetes diagnosis, years", value="", placeholder="e.g., 52"))
    hdl_c = parse_float_input(st.text_input("HDL-c, mmol/L", value="", placeholder="e.g., 1.14"))

with clinical_col_2:
    egfr = parse_float_input(st.text_input("eGFR, mL/min/1.73m2", value="", placeholder="e.g., 87"))
    cholesterol = parse_float_input(st.text_input("Total cholesterol, mmol/L", value="", placeholder="e.g., 4.39"))
    sex_choice = st.selectbox("Sex", options=["", "Female", "Male"], index=0)
    smoking_choice = st.selectbox(
        "Smoking status",
        options=["", "Never", "Former", "Current"],
        index=0,
    )

sex = {"Female": 0, "Male": 1}.get(sex_choice)
smoking_status = {"Never": 0, "Former": 1, "Current": 2}.get(smoking_choice)

clinical_values = {
    "age": age,
    "sex": sex,
    "HbA1c": hba1c,
    "SBP": sbp,
    "diab": diab,
    "eGFR": egfr,
    "smoking_status": smoking_status,
    "HDL.c": hdl_c,
    "cholesterol": cholesterol,
}


if st.button("Calculate risk", type="primary"):
    protein_missing = missing_labels(protein_values, protein_display_names)
    protein_invalid = invalid_labels(protein_values, protein_display_names)
    clinical_missing = missing_labels(clinical_values, clinical_fields)
    clinical_invalid = invalid_labels(clinical_values, clinical_fields)

    if protein_invalid or clinical_invalid:
        st.error(
            "Please check the following inputs. They must be numeric: "
            + ", ".join(protein_invalid + clinical_invalid)
        )
        st.stop()

    protein_complete = len(protein_missing) == 0
    clinical_complete = len(clinical_missing) == 0

    st.header("Results")

    if not protein_complete:
        st.warning(
            "The PRS model was not calculated because the following protein inputs are missing: "
            + ", ".join(protein_missing)
        )

    if protein_complete:
        prs = calculate_prs(protein_values)
        risk_prs = calculate_prs_model_risk(prs)

        st.subheader("Protein Risk Score")
        st.metric("PRS", f"{prs:.4f}")
        st.caption("PRS is calculated directly from the entered Olink NPX values without further standardization.")

        st.subheader("PRS Model")
        st.success(f"Predicted 10-year CVD risk: {risk_prs * 100:.2f}%")

        if clinical_complete:
            risk_clinical_prs = calculate_clinical_prs_model_risk(
                prs=prs,
                age=age,
                sex=sex,
                hba1c=hba1c,
                sbp=sbp,
                diab=diab,
                egfr=egfr,
                smoking_status=smoking_status,
                hdl_c=hdl_c,
                cholesterol=cholesterol,
            )

            st.subheader("Clinical-PRS Model")
            st.success(f"Predicted 10-year CVD risk: {risk_clinical_prs * 100:.2f}%")

            with st.expander("Show standardized clinical values"):
                st.write(f"Age standardized = {standardize(age, 'age'):.4f}")
                st.write(f"SBP standardized = {standardize(sbp, 'SBP'):.4f}")
                st.write(f"Age at diabetes diagnosis standardized = {standardize(diab, 'diab'):.4f}")
                st.write(f"HDL-c standardized = {standardize(hdl_c, 'HDL.c'):.4f}")
                st.write(f"Total cholesterol standardized = {standardize(cholesterol, 'cholesterol'):.4f}")
                st.write(f"HbA1c standardized = {standardize(hba1c, 'HbA1c'):.4f}")
                st.write(f"eGFR standardized = {standardize(egfr, 'eGFR'):.4f}")
        else:
            st.warning(
                "The Clinical-PRS model was not calculated because the following clinical inputs are missing: "
                + ", ".join(clinical_missing)
            )

    if not protein_complete and clinical_complete:
        st.info(
            "Clinical variables alone are not sufficient for the models currently implemented. "
            "A clinical-only model is not included in this calculator."
        )
