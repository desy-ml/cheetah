from ocelot import (
    Aperture,
    Cavity,
    Drift,
    Hcor,
    Marker,
    Monitor,
    Quadrupole,
    SBend,
    Solenoid,
    TDCavity,
    Vcor,
)

# Drifts
drift_arlisolg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISOLG1")
drift_arlimsog1p = Drift(l=0.1659, eid="Drift_ARLIMSOG1p")
drift_arlimcvg1 = Drift(l=0.18949999999999995, eid="Drift_ARLIMCVG1")
drift_arlibscl1 = Drift(l=0.14449999999999996, eid="Drift_ARLIBSCL1")
drift_arlibaml1 = Drift(l=0.3025, eid="Drift_ARLIBAML1")
drift_arlibscx1 = Drift(l=0.18999999999999995, eid="Drift_ARLIBSCX1")
drift_arlislhg1 = Drift(l=0.4884000000000001, eid="Drift_ARLISLHG1")
drift_arlimcvg2 = Drift(l=0.06200000000000022, eid="Drift_ARLIMCVG2")
drift_arlibcmg1 = Drift(l=0.3274999999999997, eid="Drift_ARLIBCMG1")
drift_arlibscr1 = Drift(l=0.1474700000000002, eid="Drift_ARLIBSCR1")
drift_arlirsbl1 = Drift(l=0.021380000000000118, eid="Drift_ARLIRSBL1")
drift_arlimcvg3 = Drift(l=0.29755000000000015, eid="Drift_ARLIMCVG3")
drift_arlibscr2 = Drift(l=0.18699999999999917, eid="Drift_ARLIBSCR2")
drift_arlimchm1 = Drift(l=0.41300000000000003, eid="Drift_ARLIMCHM1")
drift_arlibpmg1 = Drift(l=0.30700000000000016, eid="Drift_ARLIBPMG1")
drift_arlimcvm1 = Drift(l=0.09497000000000044, eid="Drift_ARLIMCVM1")
drift_arlirsbl2 = Drift(l=0.02137999999999923, eid="Drift_ARLIRSBL2")
drift_arlimcvg4 = Drift(l=0.29754999999999837, eid="Drift_ARLIMCVG4")
drift_arlibscr3 = Drift(l=0.18700000000000094, eid="Drift_ARLIBSCR3")
drift_arlimcvm2 = Drift(l=0.09496000000000017, eid="Drift_ARLIMCVM2")
drift_areasola1 = Drift(l=0.17503999999999914, eid="Drift_AREASOLA1")
drift_areamqzm1 = Drift(l=0.4280000000000007, eid="Drift_AREAMQZM1")
drift_areamqzm2 = Drift(l=0.20399999999999857, eid="Drift_AREAMQZM2")
drift_areamcvm1 = Drift(l=0.20400000000000035, eid="Drift_AREAMCVM1")
drift_areamqzm3 = Drift(l=0.179, eid="Drift_AREAMQZM3")
drift_areamchm1 = Drift(l=0.45000000000000084, eid="Drift_AREAMCHM1")
drift_areabscr1 = Drift(l=1.3800000000000008, eid="Drift_AREABSCR1")
drift_areaecha1 = Drift(l=0.369999999999999, eid="Drift_AREAECHA1")
drift_areamchm2 = Drift(l=0.22999999999999998, eid="Drift_AREAMCHM2")
drift_areamcvm2 = Drift(l=0.07289999999999865, eid="Drift_AREAMCVM2")
drift_armrsolt1 = Drift(l=0.17360000000000197, eid="Drift_ARMRSOLT1")
drift_armrmchm1 = Drift(l=0.3745000000000007, eid="Drift_ARMRMCHM1")
drift_armrmcvm1 = Drift(l=0.3729999999999991, eid="Drift_ARMRMCVM1")
drift_armrbpmg1 = Drift(l=0.3339999999999996, eid="Drift_ARMRBPMG1")
drift_armrmqzm1 = Drift(l=0.24900000000000028, eid="Drift_ARMRMQZM1")
drift_armrmcvm2 = Drift(l=0.359999999999999, eid="Drift_ARMRMCVM2")
drift_armrmchm2 = Drift(l=0.27000000000000113, eid="Drift_ARMRMCHM2")
drift_armrbscr1 = Drift(l=0.17899999999999844, eid="Drift_ARMRBSCR1")
drift_armrmchm3 = Drift(l=0.1498000000000006, eid="Drift_ARMRMCHM3")
drift_armrbcmg1 = Drift(l=0.15019999999999967, eid="Drift_ARMRBCMG1")
drift_armrmcvm3 = Drift(l=0.30500000000000127, eid="Drift_ARMRMCVM3")
drift_armrmqzm3 = Drift(l=0.2169999999999987, eid="Drift_ARMRMQZM3")
drift_armrbaml1 = Drift(l=0.14300000000000224, eid="Drift_ARMRBAML1")
drift_armrmcvm4 = Drift(l=0.15899999999999692, eid="Drift_ARMRMCVM4")
drift_armrtorf1 = Drift(l=0.13100000000000178, eid="Drift_ARMRTORF1")
drift_armrmqzm4 = Drift(l=0.1494999999999997, eid="Drift_ARMRMQZM4")
drift_ardgsolo1 = Drift(l=1.5455000000000005, eid="Drift_ARDGSOLO1")
drift_armrsolb1 = Drift(l=0.2259999999999991, eid="Drift_ARMRSOLB1")
drift_armrbpmg3 = Drift(l=0.3770000000000024, eid="Drift_ARMRBPMG3")
drift_armrmqzm5 = Drift(l=0.24899999999999672, eid="Drift_ARMRMQZM5")
drift_armrmcvm5 = Drift(l=0.2800000000000007, eid="Drift_ARMRMCVM5")
drift_armrmqzm6 = Drift(l=0.32400000000000156, eid="Drift_ARMRMQZM6")
drift_arbcsolc = Drift(l=0.265, eid="Drift_ARBCSOLC")
drift_arbcmbhb1 = Drift(l=0.6000000000000003, eid="Drift_ARBCMBHB1")
drift_arbcmbhb2 = Drift(l=0.5611999999999989, eid="Drift_ARBCMBHB2")
drift_arbcbpml1 = Drift(l=0.602800000000002, eid="Drift_ARBCBPML1")
drift_arbcslhb1 = Drift(l=0.38400000000000034, eid="Drift_ARBCSLHB1")
drift_arbcslhs1 = Drift(l=0.6139999999999972, eid="Drift_ARBCSLHS1")
drift_arbcbsce1 = Drift(l=0.5540000000000015, eid="Drift_ARBCBSCE1")
drift_arbcmbhb4 = Drift(l=0.26499999999999646, eid="Drift_ARBCMBHB4")
drift_ardlsolm1 = Drift(l=0.19700000000000073, eid="Drift_ARDLSOLM1")
drift_ardlmcvm1 = Drift(l=0.15900000000000403, eid="Drift_ARDLMCVM1")
drift_ardltorf1 = Drift(l=0.13099999999999823, eid="Drift_ARDLTORF1")
drift_ardlmqzm1 = Drift(l=0.36399999999999716, eid="Drift_ARDLMQZM1")
drift_ardlbpmg1 = Drift(l=0.3340000000000031, eid="Drift_ARDLBPMG1")
drift_ardlmqzm2 = Drift(l=0.341000000000001, eid="Drift_ARDLMQZM2")
drift_ardlbscr1 = Drift(l=0.5300000000000011, eid="Drift_ARDLBSCR1")
drift_ardlrxbd1 = Drift(l=0.0899999999999963, eid="Drift_ARDLRXBD1")
drift_ardlrxbd2 = Drift(l=0.6499999999999986, eid="Drift_ARDLRXBD2")
drift_ardlbsce1 = Drift(l=0.5350000000000037, eid="Drift_ARDLBSCE1")
drift_ardlbpmg2 = Drift(l=0.4749999999999994, eid="Drift_ARDLBPMG2")
drift_ardlmqzm3 = Drift(l=0.6089999999999998, eid="Drift_ARDLMQZM3")
drift_ardlmqzm4 = Drift(l=0.14949999999999614, eid="Drift_ARDLMQZM4")
drift_arshsolh1 = Drift(l=0.8702372828930167, eid="Drift_ARSHSOLH1")
drift_arshmbho1 = Drift(l=1.1127372828930164, eid="Drift_ARSHMBHO1")
drift_arshbsce2 = Drift(l=-0.8693559999999962, eid="Drift_ARSHBSCE2")
drift_arshbsce1 = Drift(l=0.03921499999999867, eid="Drift_ARSHBSCE1")
drift_arsheolh1 = Drift(l=0.887640999999995, eid="Drift_ARSHEOLH1")

# Quadrupoles
areamqzm1 = Quadrupole(l=0.122, eid="AREAMQZM1")
areamqzm2 = Quadrupole(l=0.122, eid="AREAMQZM2")
areamqzm3 = Quadrupole(l=0.122, eid="AREAMQZM3")
armrmqzm1 = Quadrupole(l=0.122, eid="ARMRMQZM1")
armrmqzm2 = Quadrupole(l=0.122, eid="ARMRMQZM2")
armrmqzm3 = Quadrupole(l=0.122, eid="ARMRMQZM3")
armrmqzm4 = Quadrupole(l=0.122, eid="ARMRMQZM4")
armrmqzm5 = Quadrupole(l=0.122, eid="ARMRMQZM5")
armrmqzm6 = Quadrupole(l=0.122, eid="ARMRMQZM6")
ardlmqzm1 = Quadrupole(l=0.122, eid="ARDLMQZM1")
ardlmqzm2 = Quadrupole(l=0.122, eid="ARDLMQZM2")
ardlmqzm3 = Quadrupole(l=0.122, eid="ARDLMQZM3")
ardlmqzm4 = Quadrupole(l=0.122, eid="ARDLMQZM4")

# SBends
arbcmbhb1 = SBend(l=0.22, eid="ARBCMBHB1")
arbcmbhb2 = SBend(l=0.22, eid="ARBCMBHB2")
arbcmbhb3 = SBend(l=0.22, eid="ARBCMBHB3")
arbcmbhb4 = SBend(l=0.22, eid="ARBCMBHB4")
arshmbho1 = SBend(
    l=0.43852543421396856,
    angle=0.8203047484373349,
    e2=-0.7504915783575616,
    eid="ARSHMBHO1",
)

# Hcors
arlimcxg1a = Hcor(l=5e-05, eid="ARLIMCXG1")
arlimcxg2a = Hcor(l=5e-05, eid="ARLIMCXG2")
arlimcxg3a = Hcor(l=5e-05, eid="ARLIMCXG3")
arlimchm1 = Hcor(l=0.02, eid="ARLIMCHM1")
arlimcxg4a = Hcor(
    l=5e-05, eid="ARLIMCXG4A"
)  # I (Jan) added A and B here in order to have unique IDs
arlimchm2 = Hcor(l=0.02, eid="ARLIMCHM2")
areamchm1 = Hcor(l=0.02, eid="AREAMCHM1")
areamchm2 = Hcor(l=0.02, eid="AREAMCHM2")
armrmchm1 = Hcor(l=0.02, eid="ARMRMCHM1")
armrmchm2 = Hcor(l=0.02, eid="ARMRMCHM2")
armrmchm3 = Hcor(l=0.02, eid="ARMRMCHM3")
armrmchm4 = Hcor(l=0.02, eid="ARMRMCHM4")
armrmchm5 = Hcor(l=0.02, eid="ARMRMCHM5")
ardlmchm1 = Hcor(l=0.02, eid="ARDLMCHM1")
ardlmchm2 = Hcor(l=0.02, eid="ARDLMCHM2")

# Vcors
arlimcxg1b = Vcor(l=5e-05, eid="ARLIMCXG1")
arlimcxg2b = Vcor(l=5e-05, eid="ARLIMCXG2")
arlimcxg3b = Vcor(l=5e-05, eid="ARLIMCXG3")
arlimcvm1 = Vcor(l=0.02, eid="ARLIMCVM1")
arlimcxg4b = Vcor(
    l=5e-05, eid="ARLIMCXG4B"
)  # I (Jan) added A and B here in order to have unique IDs
arlimcvm2 = Vcor(l=0.02, eid="ARLIMCVM2")
areamcvm1 = Vcor(l=0.02, eid="AREAMCVM1")
areamcvm2 = Vcor(l=0.02, eid="AREAMCVM2")
armrmcvm1 = Vcor(l=0.02, eid="ARMRMCVM1")
armrmcvm2 = Vcor(l=0.02, eid="ARMRMCVM2")
armrmcvm3 = Vcor(l=0.02, eid="ARMRMCVM3")
armrmcvm4 = Vcor(l=0.02, eid="ARMRMCVM4")
armrmcvm5 = Vcor(l=0.02, eid="ARMRMCVM5")
ardlmcvm1 = Vcor(l=0.02, eid="ARDLMCVM1")
ardlmcvm2 = Vcor(l=0.02, eid="ARDLMCVM2")

# Cavitys
arlirsbl1 = Cavity(l=4.139, freq=2998000000.0, eid="ARLIRSBL1")
arlirsbl2 = Cavity(l=4.139, freq=2998000000.0, eid="ARLIRSBL2")

# TDCavitys
ardlrxbd1 = TDCavity(
    l=1.0, freq=11995200000.0, tilt=1.5707963267948966, eid="ARDLRXBD1"
)
ardlrxbd2 = TDCavity(
    l=1.0, freq=11995200000.0, tilt=1.5707963267948966, eid="ARDLRXBD2"
)

# Solenoids
arlimsog1a = Solenoid(l=0.09, eid="ARLIMSOG1")
arlimsog1b = Solenoid(l=0.09, eid="ARLIMSOG1")

# Monitors
arlibscl1 = Monitor(eid="ARLIBSCL1")
arlibaml1 = Monitor(eid="ARLIBAML1")
arlibscx1 = Monitor(eid="ARLIBSCX1")
arlibcmg1 = Monitor(eid="ARLIBCMG1")
arlibscr1 = Monitor(eid="ARLIBSCR1")
arlibscr2 = Monitor(eid="ARLIBSCR2")
arlibpmg1 = Monitor(eid="ARLIBPMG1")
arlibscr3 = Monitor(eid="ARLIBSCR3")
arlibpmg2 = Monitor(eid="ARLIBPMG2")
areabscr1 = Monitor(eid="AREABSCR1")
areaecha1 = Monitor(eid="AREAECHA1")
armrbpmg1 = Monitor(eid="ARMRBPMG1")
armrbscr1 = Monitor(eid="ARMRBSCR1")
armrbcmg1 = Monitor(eid="ARMRBCMG1")
armrbpmg2 = Monitor(eid="ARMRBPMG2")
armrbaml1 = Monitor(eid="ARMRBAML1")
armrtorf1 = Monitor(eid="ARMRTORF1")
armrbscr2 = Monitor(eid="ARMRBSCR2")
armrbpmg3 = Monitor(eid="ARMRBPMG3")
armrbscr3 = Monitor(eid="ARMRBSCR3")
arbcbpml1 = Monitor(eid="ARBCBPML1")
arbcbsce1 = Monitor(eid="ARBCBSCE1")
ardltorf1 = Monitor(eid="ARDLTORF1")
ardlbpmg1 = Monitor(eid="ARDLBPMG1")
ardlbscr1 = Monitor(eid="ARDLBSCR1")
ardlbsce1 = Monitor(eid="ARDLBSCE1")
ardlbpmg2 = Monitor(eid="ARDLBPMG2")
arshbsce2 = Monitor(eid="ARSHBSCE2")
arshbsce1 = Monitor(eid="ARSHBSCE1")

# Markers
arlisolg1 = Marker(eid="ARLISOLG1")
arlieolg1 = Marker(eid="ARLIEOLG1")
arlisols1 = Marker(eid="ARLISOLS1")
arlieols1 = Marker(eid="ARLIEOLS1")
areasola1 = Marker(eid="AREASOLA1")
areaeola1 = Marker(eid="AREAEOLA1")
armrsolt1 = Marker(eid="ARMRSOLT1")
armreolt1 = Marker(eid="ARMREOLT1")
ardgsolo1 = Marker(eid="ARDGSOLO1")
ardgeolo1 = Marker(eid="ARDGEOLO1")
armrsolb1 = Marker(eid="ARMRSOLB1")
armreolb1 = Marker(eid="ARMREOLB1")
arbcsolc = Marker(eid="ARBCSOLC")
arbceolc = Marker(eid="ARBCEOLC")
ardlsolm1 = Marker(eid="ARDLSOLM1")
ardleolm1 = Marker(eid="ARDLEOLM1")
arshsolh1 = Marker(eid="ARSHSOLH1")
arsheolh1 = Marker(eid="ARSHEOLH1")
arsheolh2 = Marker(eid="ARSHEOLH2")

# Apertures
arlislhg1 = Aperture(eid="ARLISLHG1")
arbcslhb1 = Aperture(eid="ARBCSLHB1")
arbcslhs1 = Aperture(eid="ARBCSLHS1")

# Lattice
cell = (
    arlisolg1,
    drift_arlisolg1,
    arlimsog1a,
    arlimsog1b,
    drift_arlimsog1p,
    arlimcxg1a,
    arlimcxg1b,
    drift_arlimcvg1,
    arlibscl1,
    drift_arlibscl1,
    arlibaml1,
    drift_arlibaml1,
    arlibscx1,
    drift_arlibscx1,
    arlislhg1,
    drift_arlislhg1,
    arlimcxg2a,
    arlimcxg2b,
    drift_arlimcvg2,
    arlibcmg1,
    drift_arlibcmg1,
    arlibscr1,
    drift_arlibscr1,
    arlieolg1,
    arlisols1,
    arlirsbl1,
    drift_arlirsbl1,
    arlimcxg3a,
    arlimcxg3b,
    drift_arlimcvg3,
    arlibscr2,
    drift_arlibscr2,
    arlimchm1,
    drift_arlimchm1,
    arlibpmg1,
    drift_arlibpmg1,
    arlimcvm1,
    drift_arlimcvm1,
    arlirsbl2,
    drift_arlirsbl2,
    arlimcxg4a,
    arlimcxg4b,
    drift_arlimcvg4,
    arlibscr3,
    drift_arlibscr3,
    arlimchm2,
    drift_arlimchm1,
    arlibpmg2,
    drift_arlibpmg1,
    arlimcvm2,
    drift_arlimcvm2,
    arlieols1,
    areasola1,
    drift_areasola1,
    areamqzm1,
    drift_areamqzm1,
    areamqzm2,
    drift_areamqzm2,
    areamcvm1,
    drift_areamcvm1,
    areamqzm3,
    drift_areamqzm3,
    areamchm1,
    drift_areamchm1,
    areabscr1,
    drift_areabscr1,
    areaecha1,
    drift_areaecha1,
    areamchm2,
    drift_areamchm2,
    areamcvm2,
    drift_areamcvm2,
    areaeola1,
    armrsolt1,
    drift_armrsolt1,
    armrmchm1,
    drift_armrmchm1,
    armrmcvm1,
    drift_armrmcvm1,
    armrbpmg1,
    drift_armrbpmg1,
    armrmqzm1,
    drift_armrmqzm1,
    armrmcvm2,
    drift_armrmcvm2,
    armrmchm2,
    drift_armrmchm2,
    armrbscr1,
    drift_armrbscr1,
    armrmqzm2,
    drift_armrmqzm1,
    armrmchm3,
    drift_armrmchm3,
    armrbcmg1,
    drift_armrbcmg1,
    armrmcvm3,
    drift_armrmcvm3,
    armrbpmg2,
    drift_armrbpmg1,
    armrmqzm3,
    drift_armrmqzm3,
    armrbaml1,
    drift_armrbaml1,
    armrmcvm4,
    drift_armrmcvm4,
    armrtorf1,
    drift_armrtorf1,
    armrmchm4,
    drift_armrmchm2,
    armrbscr2,
    drift_armrbscr1,
    armrmqzm4,
    drift_armrmqzm4,
    armreolt1,
    ardgsolo1,
    drift_ardgsolo1,
    ardgeolo1,
    armrsolb1,
    drift_armrsolb1,
    armrbpmg3,
    drift_armrbpmg3,
    armrmqzm5,
    drift_armrmqzm5,
    armrmcvm5,
    drift_armrmcvm5,
    armrmchm5,
    drift_armrmchm2,
    armrbscr3,
    drift_armrbscr1,
    armrmqzm6,
    drift_armrmqzm6,
    armreolb1,
    arbcsolc,
    drift_arbcsolc,
    arbcmbhb1,
    drift_arbcmbhb1,
    arbcmbhb2,
    drift_arbcmbhb2,
    arbcbpml1,
    drift_arbcbpml1,
    arbcslhb1,
    drift_arbcslhb1,
    arbcslhs1,
    drift_arbcslhs1,
    arbcbsce1,
    drift_arbcbsce1,
    arbcmbhb3,
    drift_arbcmbhb1,
    arbcmbhb4,
    drift_arbcmbhb4,
    arbceolc,
    ardlsolm1,
    drift_ardlsolm1,
    ardlmcvm1,
    drift_ardlmcvm1,
    ardltorf1,
    drift_ardltorf1,
    ardlmchm1,
    drift_armrmqzm1,
    ardlmqzm1,
    drift_ardlmqzm1,
    ardlbpmg1,
    drift_ardlbpmg1,
    ardlmqzm2,
    drift_ardlmqzm2,
    ardlbscr1,
    drift_ardlbscr1,
    ardlrxbd1,
    drift_ardlrxbd1,
    ardlrxbd2,
    drift_ardlrxbd2,
    ardlbsce1,
    drift_ardlbsce1,
    ardlbpmg2,
    drift_ardlbpmg2,
    ardlmcvm2,
    drift_armrmqzm1,
    ardlmqzm3,
    drift_ardlmqzm3,
    ardlmchm2,
    drift_armrmqzm1,
    ardlmqzm4,
    drift_ardlmqzm4,
    ardleolm1,
    arshsolh1,
    drift_arshsolh1,
    arshmbho1,
    drift_arshmbho1,
    arshbsce2,
    drift_arshbsce2,
    arshbsce1,
    drift_arshbsce1,
    arsheolh1,
    drift_arsheolh1,
    arsheolh2,
)
