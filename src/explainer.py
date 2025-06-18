import shap, warnings
import matplotlib.pyplot as plt
import inspect

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
    try:
        explainer = _choose_explainer(model, X)
        # Check if 'check_additivity' is a valid argument for the explainer's __call__
        call_args = inspect.signature(explainer.__call__).parameters
        if "check_additivity" in call_args:
            shap_values = explainer(X, check_additivity=False)
        else:
            shap_values = explainer(X)

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
    except Exception as e:
        print(f"⚠️  SHAP explainability failed: {e}")
        return
