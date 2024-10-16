import os
import re

import pandas as pd
import yaml

DOCKER_MODE = os.getenv("DOCKER_MODE")
try:
    if DOCKER_MODE == "1":
        PATH_SAVE_QUESTION = os.path.join(os.getcwd(), "ref_question/")
    elif (DOCKER_MODE == "0") or (DOCKER_MODE is None):
        PATH_SAVE_QUESTION = os.path.join(os.getcwd(), "src/maif_corpus/ref_question/")
    elif (DOCKER_MODE is not None) and (DOCKER_MODE not in ("1", "0")):
        raise ValueError(f"Invalid value for DOCKER_MODE: {DOCKER_MODE}")
except Exception as e:
    raise RuntimeError(f"Import Default Forms KO - {e}") from e


def list_questions(path: str) -> list:
    """Lists all files in the specified directory.

    Args:
        path (str): The path to the directory.

    Returns:
        list: A list of filenames in the directory.

    Raises:
        Exception: If there is an error accessing the directory.
    """
    try:
        return os.listdir(path)
    except Exception as e:
        raise e from e


def save_questions(name: str, questions: pd.DataFrame, path: str = PATH_SAVE_QUESTION) -> tuple[str, list]:
    """Saves a DataFrame of questions to a YAML file.

    Args:
        name (str): The name of the file (without extension).
        questions (pd.DataFrame): A DataFrame containing the questions.
        path (str, optional): The directory where the file will be saved. Defaults to PATH_SAVE_QUESTION.

    Returns:
        tuple[str, list]: A success message and a list of current files in the directory.

    Raises:
        Exception: If there is an error saving the file.
    """
    try:
        data = yaml.dump(questions["Question"].to_list())
        with open(os.path.join(path, f"{name}.yaml"), "w") as file:
            file.write(data)
        return f"Votre référentiel {name} a bien été enregistré", list_questions(path)
    except Exception as e:
        raise Exception from e


def load_questions(name: str, path: str = PATH_SAVE_QUESTION) -> tuple[pd.DataFrame, list]:
    """Loads a YAML file of questions into a DataFrame.

    Args:
        name (str): The name of the file (with extension).
        path (str, optional): The directory where the file is located. Defaults to PATH_SAVE_QUESTION.

    Returns:
        tuple[pd.DataFrame, list]: A DataFrame containing the questions and a list of current files in the directory.

    Raises:
        ValueError: If the file does not exist.
        Exception: If there is an error loading the file.
    """
    if name in os.listdir(path):
        try:
            with open(os.path.join(path, name)) as file:
                Questions = yaml.safe_load(file)
            Questions = pd.DataFrame([k for k in Questions], columns=["Question"])
            return Questions, list_questions(path)
        except Exception as e:
            raise RuntimeError(f"Fail to load {e}") from e
    else:
        raise ValueError("Ce référentiel n'existe pas")


def delete_questions(name: str, path: str = PATH_SAVE_QUESTION) -> tuple[str, list]:
    """Deletes a YAML file of questions.

    Args:
        name (str): The name of the file (with or without extension).
        path (str, optional): The directory where the file is located. Defaults to PATH_SAVE_QUESTION.

    Returns:
        tuple[str, list]: A success message and a list of current files in the directory.

    Raises:
        ValueError: If the file does not exist or is not a valid file.
        Exception: If there is an error deleting the file.
    """
    if not re.search(".yaml$", name):
        name = name + ".yaml"
    if name in os.listdir(path):
        try:
            file = os.path.join(path, name)
            if os.path.isfile(file):
                os.remove(file)
                return f"Le référentiel {name} a bien été supprimé", list_questions(path)
            else:
                raise ValueError(f"Ceci : {name} n'est pas un référentiel")
        except Exception as e:
            raise f"Fail to load {e}" from e
    else:
        raise ValueError(f"Le référentiel {name} n'existe pas")


class Questions:
    """Class to manage questions saved in YAML files.

    Attributes:
        path (str): The directory where question files are stored.
        state (list): A list of current files in the directory.
    """

    def __init__(self, path: str) -> None:
        """Initializes the Questions class with a specific directory path.

        Args:
            path (str): The directory where question files are stored.
        """
        self.path = path
        self.state = list_questions(path)

    def list_questions(self) -> list:
        """Lists all question files in the directory.

        Returns:
            list: A list of filenames in the directory.
        """
        return list_questions(self.path)

    def save_questions(self, name: str, questions: pd.DataFrame) -> str:
        """Saves a DataFrame of questions to a YAML file.

        Args:
            name (str): The name of the file (without extension).
            questions (pd.DataFrame): A DataFrame containing the questions.

        Returns:
            str: A success message.
        """
        res, self.state = save_questions(name, questions, self.path)
        return res

    def load_questions(self, name: str) -> pd.DataFrame:
        """Loads a YAML file of questions into a DataFrame.

        Args:
            name (str): The name of the file (with extension).

        Returns:
            pd.DataFrame: A DataFrame containing the questions.
        """
        res, self.state = load_questions(name, self.path)
        return res

    def delete_questions(self, name: str) -> str:
        """Deletes a YAML file of questions.

        Args:
            name (str): The name of the file (with or without extension).

        Returns:
            str: A success message.
        """
        res, self.state = delete_questions(name, self.path)
        return res
