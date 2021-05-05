FROM python:3.7

RUN apt-get update && apt-get install -y -q \
    mpich \
    libmpich-dev \
    libgl1-mesa-glx

ADD ./requirements.txt /app/requirements.txt
ADD ./requirements-optional.txt /app/requirements-optional.txt
ADD ./requirements-mpi.txt /app/requirements-mpi.txt

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir wheel && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements-optional.txt && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements-mpi.txt && \
    rm -rf ~/.cache/pip

ADD ./devito /app/devito
ADD ./tests /app/tests
ADD ./examples /app/examples
ADD ./benchmarks /app/benchmarks
COPY setup.cfg /app/

ADD docker/run-jupyter.sh /jupyter
ADD docker/run-tests.sh /tests
ADD docker/run-print-defaults.sh /print-defaults
ADD docker/entrypoint.sh /docker-entrypoint.sh

RUN chmod +x \
    /print-defaults \
    /jupyter \
    /tests \
    /docker-entrypoint.sh

WORKDIR /app

ENV DEVITO_ARCH="gcc"
ENV DEVITO_LANGUAGE="openmp"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]
