```
                                                ,@,
                                               ,@@@,
                                              ,@@@@@,
                                       `@@@@@@@@@@@@@@@@@@@`
                                         `@@@@@@@@@@@@@@@`
                                           `@@@@@@@@@@@`
                                          ,@@@@@@`@@@@@@,
                                          @@@@`     `@@@@
                                         ;@`           `@;
 _______  _______  ___      ___   _______  _______  ___   __   __  _______  ___   _  _______  _______
|       ||       ||   |    |   | |       ||       ||   | |  | |  ||   _   ||   | | ||   _   ||       |
|    _  ||   _   ||   |    |   | |_     _||  _____||   | |  |_|  ||  |_|  ||   |_| ||  |_|  ||_     _|
|   |_| ||  | |  ||   |    |   |   |   |  | |_____ |   | |       ||       ||      _||       |  |   |
|    ___||  |_|  ||   |___ |   |   |   |  |_____  ||   | |_     _||       ||     |_ |       |  |   |
|   |    |       ||       ||   |   |   |   _____| ||   |   |   |  |   _   ||    _  ||   _   |  |   |
|___|    |_______||_______||___|   |___|  |_______||___|   |___|  |__| |__||___| |_||__| |__|  |___|
```

PolitsiyaKAT sends misbehaving antennas to icy northern isolation.

This is a crude automated flagger that finds problems in calibrated data such as phase and amplitude drifts. It then flags affected baselines based on user supplied clipping criteria. Currently we apply this tool to automatic and 
semi-automatic reductions of MeerKAT and VLA data.

I hope it will be useful to the astronomy community to make this laborous task easier. Contributions are very
welcome.

#Installation
We recommend you use a virtual environment 
```
$virtualenv politsiyakat
$source politsiyakat/bin/activate
$pip install politsiyakat
```
