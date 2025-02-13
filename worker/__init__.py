from celery import Celery

app = Celery("worker")
app.config_from_object("worker.celeryconfig")
print("Registered tasks:", app.tasks.keys())