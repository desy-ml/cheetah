from cheetah import Segment, Drift, Undulator, BPM, HorizontalCorrector, VerticalCorrector, ParticleBeam

def test_segment_code():
    segment = Segment([BPM(name="BPM1SMATCH"),
                       Drift(length=1.0),
                       BPM(name="BPM6SMATCH"),
                       Drift(length=1.0),
                       VerticalCorrector(length=0.3, name="V7SMATCH"),
                       Drift(length=0.2),
                       HorizontalCorrector(length=0.3, name="H10SMATCH"),
                       Drift(length=7.0),
                       HorizontalCorrector(length=0.3, name="H12SMATCH"),
                       Drift(length=0.05),
                       BPM(name="BPM13SMATCH"),
                       Drift(length=0.05),
                       VerticalCorrector(length=0.1, name="V14SMATCH"),
                       Drift(length=3.5),
                       BPM(name="BPM14SMATCH"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND1"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND2"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND3"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND4"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND5"),
                       Drift(length=0.2),
                       Undulator(length=3.0),
                       Drift(length=0.2),
                       BPM(name="BPM5UND6"),
                       Drift(length=6.3),
                       Drift(length=0.01, name="GMD"),
                       Drift(length=0.2),
                       BPM(name="Photon_BPM_1"),
                       Drift(length=3.0),
                       BPM(name="Photon_BPM_2")])

    segment.BPM13SMATCH.is_active = True

    assert str(segment) == str(Segment([BPM(name="BPM1SMATCH"), Drift(length=1.00, name="Drift_000001"), BPM(name="BPM6SMATCH"), \
           Drift(length=1.00, name="Drift_000003"), VerticalCorrector(length=0.30, angle=0.0, name="V7SMATCH"), Drift(length=0.20, \
           name="Drift_000005"), HorizontalCorrector(length=0.30, angle=0.0, name="H10SMATCH"), Drift(length=7.00, name="Drift_000007"), \
           HorizontalCorrector(length=0.30, angle=0.0, name="H12SMATCH"), Drift(length=0.05, name="Drift_000009"), BPM(name="BPM13SMATCH"), \
           Drift(length=0.05, name="Drift_000011"), VerticalCorrector(length=0.10, angle=0.0, name="V14SMATCH"), \
           Drift(length=3.50, name="Drift_000013"), BPM(name="BPM14SMATCH"), Drift(length=0.20, name="Drift_000015"), \
           Undulator(length=3.00, name="Undulator_000016"), Drift(length=0.20, name="Drift_000017"), BPM(name="BPM5UND1"), \
           Drift(length=0.20, name="Drift_000019"), Undulator(length=3.00, name="Undulator_000020"), Drift(length=0.20, name="Drift_000021"), \
           BPM(name="BPM5UND2"), Drift(length=0.20, name="Drift_000023"), Undulator(length=3.00, name="Undulator_000024"), \
           Drift(length=0.20, name="Drift_000025"), BPM(name="BPM5UND3"), Drift(length=0.20, name="Drift_000027"), \
           Undulator(length=3.00, name="Undulator_000028"), Drift(length=0.20, name="Drift_000029"), BPM(name="BPM5UND4"), \
           Drift(length=0.20, name="Drift_000031"), Undulator(length=3.00, name="Undulator_000032"), \
           Drift(length=0.20, name="Drift_000033"), BPM(name="BPM5UND5"), Drift(length=0.20, name="Drift_000035"), \
           Undulator(length=3.00, name="Undulator_000036"), Drift(length=0.20, name="Drift_000037"), BPM(name="BPM5UND6"), \
           Drift(length=6.30, name="Drift_000039"), Drift(length=0.01, name="GMD"), Drift(length=0.20, name="Drift_000041"), \
           BPM(name="Photon_BPM_1"), Drift(length=3.00, name="Drift_000043"), BPM(name="Photon_BPM_2")]))
