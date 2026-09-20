import app
app.demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7861)
