FROM kernsuite/base:2
MAINTAINER bhugo@ska.ac.za

RUN apt-get install -y software-properties-common
RUN apt-add-repository -s ppa:kernsuite/kern-2
RUN apt-add-repository multiverse
RUN apt-add-repository restricted
RUN apt-get update
RUN apt-get install -y python-casacore
RUN apt-get install -y python-pip
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libfreetype6-dev
RUN ln -s /usr/include/freetype2/ft2build.h /usr/include/ #bug in matplotlib
RUN apt-get install -y libpng-dev
RUN pip install pip setuptools wheel -U

ADD politsiyakat /src/politsiyakat/politsiyakat
ADD MANIFEST.in /src/politsiyakat/MANIFEST.in
ADD requirements.txt /src/politsiyakat/requirements.txt
ADD setup.py /src/politsiyakat/setup.py
ADD README.md /src/politsiyakat/README.md

RUN pip install /src/politsiyakat

ENTRYPOINT ["python", "-m", "politsiyakat"]
CMD ["--help"]