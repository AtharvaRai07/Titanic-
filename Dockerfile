FROM astrocrpublic.azurecr.io/runtime:3.1-14

RUN pip install apache-airflow-providers-google apache-airflow-providers-postgres

