ARG PYTHON_BASE=3.11.4-slim
# build stage
FROM python:$PYTHON_BASE AS builder


# install PDM
RUN pip install -U pdm
# disable update check
ENV PDM_CHECK_UPDATE=false
# copy files
COPY pyproject.toml pdm.lock /project/
# COPY . /project/


# install dependencies and project into the local packages directory
WORKDIR /project

RUN pdm install --check --prod --no-editable

# RUN pdm run python -m spacy download ru_core_news_sm

# run stage
FROM python:$PYTHON_BASE

# retrieve packages from build stage
COPY --from=builder /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"
# set command/entrypoint, adapt to fit your needs

RUN apt-get update && apt-get install -y python3-opencv ffmpeg
RUN pip install opencv-python

COPY . /project/

# RUN cd /project


CMD ["streamlit", "run", "/project/hack/main.py", "--server.port=8501", "--server.address=0.0.0.0"]