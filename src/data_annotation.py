from pathlib import Path
from itertools import combinations
from typing import List, Dict, Union
import os
import json

from dotenv import load_dotenv, find_dotenv
from rapidata import RapidataClient

load_dotenv(find_dotenv())

CLIENT_ID = os.getenv("CLIENT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# function to retrieve the images
def get_prompts(
        prompt_file_name: str
) -> Dict:
    """
    Retrieves prompts from a JSON file.

    Args:
        prompt_file_name (str): Name of the JSON file containing prompts.

    Returns:
        Dict: A dictionary containing prompts.
    """
    curr_path = Path(__file__).parent.parent
    prompt_file_path = curr_path / "data" / prompt_file_name
    
    with open(prompt_file_path, "r") as f:
        prompts = json.load(f)
    
    return prompts

def get_image_prompt_pairs(
        model_names: List[str]
) -> Union[List[List[str]], List[str]]:
    """
    Gets image and prompt pairs for the specified models to pass to Rapidata order.

    Args:
        model_names (List[str]): List of model names.
    
    Returns:
        Union[List[List[str]], List[str]]:
            - List of image paths.
            - List of prompts.
    """
    curr_path = Path(__file__).parent.parent
    images_folder = curr_path / "images"

    # assuming the file names in the model directories are the same
    image_names = [str(image_name).split("/")[-1].split("_")[-1] for image_name in images_folder.iterdir() if image_name.is_file()]
    
    prompts = get_prompts("prompts.json")

    comparisons = []
    prompt_list = []
    # for each image type, we create a comparison between two models
    for image in image_names:
        for model1, model2 in combinations(model_names, 2):
            path_image1 = images_folder / f"{model1}_{image}"
            path_image2 = images_folder / f"{model2}_{image}"
            comparisons.append([str(path_image1), str(path_image2)])
            prompt_list.append(prompts[image[:-4]]) # remove file format name
    
    return comparisons, prompt_list

def get_image_preference(
        rapi: RapidataClient,
        prompts: List[str],
        images: List[List[str]],
        responses_per_datapoint: int = 5,
        task_name: str = "Image Prompt Alignment Test",
        instruction: str = "Which image follows the prompt more accurately?",
):
    """
    Creates a Rapidata order for image preference testing.

    Args:
        rapi (RapidataClient): Rapidata client instance.
        prompts (List[str]): List of prompts.
        images (List[List[str]]): List of image pairs.
        responses_per_datapoint (int): Number of responses per datapoint.
        task_name (str): Name of the task.
        instruction (str): Instruction for the task.
    
    Returns:
        RapidataOrder: Rapidata order.
    """
    order = rapi.order.create_compare_order(
        name=task_name,
        instruction=instruction,
        responses_per_datapoint=responses_per_datapoint,
        datapoints=images,
        contexts=prompts
    ).run()

    return order

if __name__ == "__main__":
    # get path to data folder
    curr_path = Path(__file__).parent.parent
    data_folder = curr_path / "data"

    rapi = RapidataClient(client_id=CLIENT_ID, client_secret=ACCESS_TOKEN)
    images, prompts = get_image_prompt_pairs(["4o", "dalle3", "halfmoon", "stablediffusion"])
    order = get_image_preference(rapi, prompts, images)
    order.display_progress_bar()
    results = order.get_results()
    results.to_json(f"{data_folder}/results.json")
    print("Results saved to results.json")