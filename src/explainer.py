import shap, warnings
import matplotlib.pyplot as plt

def _choose_explainer(model, X):
    """
    Pick the most appropriate SHAP explainer based on model class.
    """
    mtype = type(model).__name__.lower()

    if "xgb" in mtype or "lgbm" in mtype or "forest" in mtype or "tree" in mtype:
        return shap.TreeExplainer(model)
    if "logistic" in mtype or ("svc" in mtype and getattr(model, "kernel", "") == "linear"):
        return shap.LinearExplainer(model, X)
    if "linearregression" in mtype:
        return shap.LinearExplainer(model, X)

    # Fallback – can be slow
    warnings.warn("Using KernelExplainer (may be slow).")
    return shap.Explainer(model.predict, X)

def explain_model(model, X, feature_names, max_display=20, save_path="models/shap_summary.png"):
    """
    Compute SHAP values, plot summary, and save to disk.
    """
    print("\n🔎 Generating SHAP explanations...")

    explainer   = _choose_explainer(model, X)
    shap_values = explainer(X, check_additivity=False)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,               # don't block headless environments
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # free memory
    print(f"📊 SHAP summary plot saved → {save_path}")
