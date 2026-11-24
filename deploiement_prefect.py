from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    train_flow,
)

TZ = "Africa/Tunis"

if __name__ == "__main__":
    serve(
        # Full pipeline: every night at 02:00 (Tunis time)
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 2 * * *", timezone=TZ),
            tags=["full-pipeline", "mlops"],
        ),
        # Retraining: every Monday at 03:00
        train_flow.to_deployment(
            name="ml-pipeline-train",
            schedule=Cron("0 3 * * 1", timezone=TZ),
            tags=["training", "mlops"],
        ),
        # Prediction: every hour
        predict_flow.to_deployment(
            name="ml-pipeline-predict",
            schedule=Cron("0 * * * *", timezone=TZ),
            tags=["prediction", "mlops"],
        ),
        # Manual only (UI "Run" button or `prefect deployment run`)
        code_flow.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    )
