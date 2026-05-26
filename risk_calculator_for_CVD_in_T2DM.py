import streamlit as st
import numpy as np

# ============================================================
# 1. PRS 构建权重
# PRS 构建不含截距项
# ============================================================

protein_weights = {
    "ACAA1": -0.4026,
    "ALPP": -0.0284,
    "HMOX1": 0.4618,
    "IL18RAP": -0.1533,
    "ITGB1BP1": 0.8919,
    "MICB/MICA": 0.1887,
    "MUC16": -0.0831,
    "NT-proBNP": 0.4211,
    "REN": 0.3902,
    "TIMP3": 0.1159,
    "VIM": 0.1217,
    "XPNPEP2": 0.0954
}

# ============================================================
# 2. 训练集标准化参数
# 来自 standardization_parameters_from_training_set.csv
# ============================================================

standardization_params = {
    "age": {
        "mean": 59.793496,
        "sd": 7.108997
    },
    "SBP": {
        "mean": 144.107640,
        "sd": 17.731316
    },
    "diab": {
        "mean": 52.374729,
        "sd": 9.193202
    },
    "HDL.c": {
        "mean": 1.138337,
        "sd": 0.265358
    },
    "cholesterol": {
        "mean": 4.388545,
        "sd": 0.962797
    },
    "HbA1c": {
        "mean": 52.149975,
        "sd": 13.383983
    },
    "eGFR": {
        "mean": 87.272670,
        "sd": 7.394785
    }
}

# ============================================================
# 3. PRS model 系数
# ============================================================

prs_model_intercept = -1.6732
prs_model_beta = 1.0082

# ============================================================
# 4. Clinical-PRS model 系数
# 连续变量使用标准化后的值
# sex 和 smoking_status 直接使用编码值
# ============================================================

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
    "cholesterol": 0.0057
}

# ============================================================
# 5. 工具函数
# ============================================================

def logistic(lp):
    return 1 / (1 + np.exp(-lp))


def standardize(value, variable):
    mean = standardization_params[variable]["mean"]
    sd = standardization_params[variable]["sd"]
    return (value - mean) / sd


def calculate_prs(protein_values):
    prs = 0
    for protein, weight in protein_weights.items():
        prs += protein_values[protein] * weight
    return prs


def calculate_prs_model_risk(prs):
    lp = prs_model_intercept + prs_model_beta * prs
    risk = logistic(lp)
    return lp, risk


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
    cholesterol
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

    risk = logistic(lp)
    return lp, risk


# ============================================================
# 6. Streamlit 页面
# ============================================================

st.set_page_config(
    page_title="CVD Risk Calculator",
    layout="centered"
)

st.title("10-year CVD Risk Calculator for Patients with T2DM")

st.markdown(
    """
    This calculator estimates the predicted 10-year risk of cardiovascular disease 
    in patients with type 2 diabetes mellitus.
    """
)

st.warning(
    "This calculator is for research use only and should not replace clinical judgment."
)

# ============================================================
# 7. 输入蛋白变量
# ============================================================

st.header("Step 1. Enter protein values")

st.caption(
    "The protein risk score is calculated as the weighted sum of the 12 selected proteins without an intercept."
)

protein_values = {}

for protein in protein_weights.keys():
    protein_values[protein] = st.number_input(
        label=protein,
        value=0.0,
        format="%.4f"
    )

# ============================================================
# 8. 输入临床变量
# ============================================================

st.header("Step 2. Enter clinical variables")

age = st.number_input(
    "Age, years",
    value=60.0,
    format="%.2f"
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "0 = Female" if x == 0 else "1 = Male"
)

hba1c = st.number_input(
    "HbA1c, mmol/L",
    value=52.0,
    format="%.2f"
)

sbp = st.number_input(
    "Systolic blood pressure, mmHg",
    value=144.0,
    format="%.2f"
)

diab = st.number_input(
    "Age at diabetes diagnosis, years",
    value=52.0,
    format="%.2f"
)

egfr = st.number_input(
    "eGFR, mL/min/1.73m²",
    value=87.0,
    format="%.2f"
)

smoking_status = st.selectbox(
    "Smoking status",
    options=[0, 1, 2],
    format_func=lambda x: (
        "0 = Never" if x == 0
        else "1 = Former" if x == 1
        else "2 = Current"
    )
)

hdl_c = st.number_input(
    "HDL-c, mmol/L",
    value=1.14,
    format="%.4f"
)

cholesterol = st.number_input(
    "Total cholesterol, mmol/L",
    value=4.39,
    format="%.4f"
)

# ============================================================
# 9. 计算风险
# ============================================================

if st.button("Calculate risk"):

    prs = calculate_prs(protein_values)

    lp_prs, risk_prs = calculate_prs_model_risk(prs)

    lp_clinical_prs, risk_clinical_prs = calculate_clinical_prs_model_risk(
        prs=prs,
        age=age,
        sex=sex,
        hba1c=hba1c,
        sbp=sbp,
        diab=diab,
        egfr=egfr,
        smoking_status=smoking_status,
        hdl_c=hdl_c,
        cholesterol=cholesterol
    )

    st.header("Results")

    st.subheader("Protein Risk Score")
    st.write(f"PRS = {prs:.4f}")

    st.subheader("PRS model")
    st.write(f"Linear predictor = {lp_prs:.4f}")
    st.success(f"Predicted 10-year CVD risk = {risk_prs * 100:.2f}%")

    st.subheader("Clinical-PRS model")
    st.write(f"Linear predictor = {lp_clinical_prs:.4f}")
    st.success(f"Predicted 10-year CVD risk = {risk_clinical_prs * 100:.2f}%")

    with st.expander("Show standardized clinical values"):
        st.write(f"Age standardized = {standardize(age, 'age'):.4f}")
        st.write(f"SBP standardized = {standardize(sbp, 'SBP'):.4f}")
        st.write(f"Age at diabetes diagnosis standardized = {standardize(diab, 'diab'):.4f}")
        st.write(f"HDL-c standardized = {standardize(hdl_c, 'HDL.c'):.4f}")
        st.write(f"Total cholesterol standardized = {standardize(cholesterol, 'cholesterol'):.4f}")
        st.write(f"HbA1c standardized = {standardize(hba1c, 'HbA1c'):.4f}")
        st.write(f"eGFR standardized = {standardize(egfr, 'eGFR'):.4f}")