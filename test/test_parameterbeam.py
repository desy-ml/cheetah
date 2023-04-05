from cheetah import ParameterBeam


def test_create_from_parameters():
    _ = ParameterBeam.from_parameters()


def test_transform_to():
    dummy_parameterbeam = ParameterBeam.from_parameters()
    new_mu_x = 1e-3
    new_parambeam = dummy_parameterbeam.transformed_to(mu_x=new_mu_x)
    assert isinstance(new_parambeam, ParameterBeam)
    assert new_parambeam.mu_x == 1e-3
