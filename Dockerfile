
FROM python:3.7

LABEL maintainer="Team Lax @saliu.biswajit.jui.bimesh.taimoor"

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run app.py