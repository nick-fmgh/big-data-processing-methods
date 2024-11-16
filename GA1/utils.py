from typing import Callable, Any, Dict

def get_validated_input(
        prompt: str,
        validator: Callable[[Any], bool],
        converter: Callable[[str], Any],
        error_message: str
) -> Any:
    while True:
        try:
            value = converter(input(prompt))
            if validator(value):
                return value
            raise ValueError(error_message)
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def run_dialog(params: Dict, input_configs: Dict) -> Dict:
    for param_name, config in input_configs.items():
        params[param_name] = get_validated_input(
            config["prompt"],
            config["validator"],
            config["converter"],
            config["error_message"]
        )

    return params


def read_input_set(params:Dict, input_configs: Dict):
    for param_name, config in input_configs.items():
        params[param_name] = get_validated_input(
            "",
            config["validator"],
            config["converter"],
            config["error_message"]
        )

    return params