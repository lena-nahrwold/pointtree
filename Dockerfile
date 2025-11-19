FROM josafatburmeister/pointtorch:latest

COPY run_scripts /opt

WORKDIR /opt/

RUN apt-get update && apt-get install -y g++ vim

# RUN python -m pip install pointtree[dev,docs]
# install pre-release version of circle_detection package
RUN python -m pip install --force-reinstall git+https://github.com/josafatburmeister/circle_detection.git