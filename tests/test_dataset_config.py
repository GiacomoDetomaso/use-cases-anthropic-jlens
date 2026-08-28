def test_dataset_configuration_is_valid():
    import use_cases_anthropic_jlens.settings as settings

    assert settings.settings is not None


def test_target_class_descriptions_include_configured_and_fallback_values():
    from use_cases_anthropic_jlens.settings import settings

    target_dataset = settings.target_transformation_examples_dataset

    assert target_dataset.get_class_description("adversarial") == (
        "Attempts to make the recipient execute attacker-supplied code or commands."
    )
    assert target_dataset.get_class_description("unknown") == "no description for this class"
