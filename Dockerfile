FROM radioastro/base
MAINTAINER bhugo@ska.ac.za

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y -s ppa:radio-astro/main
RUN apt-get update

RUN apt-get install -y python-casacore
RUN apt-get install -y python-pip
RUN pip install pip -U

ADD politsiyakat /src/politsiyakat/politsiyakat
ADD MANIFEST.in /src/politsiyakat/MANIFEST.in
ADD requirements.txt /src/politsiyakat/requirements.txt
ADD setup.py /src/politsiyakat/setup.py
ADD README.md /src/politsiyakat/README.md

RUN pip install /src/politsiyakat

ENTRYPOINT ["python -m politsiyakat"]
CMD ["--help"]