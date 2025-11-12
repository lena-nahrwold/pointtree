FROM josafatburmeister/pointtorch:latest

COPY src /opt/src
COPY run_semantic_segmentation_metrics.py /opt/run_semantic_segmentation_metrics.py
COPY nearest_neighbor_alignment.py /opt/nearest_neighbor_alignment.py
COPY nearest_neighbor_ordering.py /opt/nearest_neighbor_ordering.py

WORKDIR /opt/

RUN apt-get update && apt-get install -y g++

RUN python -m pip install pointtree[dev,docs]
# install pre-release version of circle_detection package
RUN python -m pip install --force-reinstall git+https://github.com/josafatburmeister/circle_detection.git

RUN chmod a+x /opt/run_semantic_segmentation_metrics.py
RUN chmod a+x /opt/nearest_neighbor_alignment.py
RUN chmod a+x /opt/nearest_neighbor_ordering.py
