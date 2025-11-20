from celery import Celery 
from datetime import timedelta

app = Celery('tasks', broker='amqp://localhost', backend='rpc://')

app.conf.timezone = 'UTC'

@app.task
def print_hi():
    print("Hi!")

@app.task
def print_ask():
    print("How are you?")

app.conf.beat_schedule = {
    'run-every-5-seconds': {
        'task': 'tasks.print_hi',
        'schedule': timedelta(seconds=5),
    },
    'run-every-7-seconds': {
        'task': 'tasks.print_ask',
        'schedule': timedelta(seconds=7),
    },
}


