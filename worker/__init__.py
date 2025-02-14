from celery import Celery

app = Celery("worker")
app.config_from_object("config.celeryconfig")
print("Registered tasks:", app.tasks.keys())