__author__ = ("prohde", "dr-joe-wirth")
import sys

import requests


# functions
def _submit(user_name: str, repo_name: str, tag: str, access_token: str) -> None:
    """tricks Zenodo into archiving a pre-existing release

    Args:
        user_name (str): github username
        repo_name (str): github repository name
        tag (str): the desired tag to archive
        access_token (str): the access token from webhook page in repo settings
    """
    # constant
    HEADERS = {"Accept": "application/vnd.github.v3+json"}

    # build the repo
    repo = "/".join((user_name, repo_name))

    # get the repo response and the release response for the repo
    repo_response = requests.get(
        f"https://api.github.com/repos/{repo}", headers=HEADERS
    )
    release_response = requests.get(
        f"https://api.github.com/repos/{repo}/releases", headers=HEADERS
    )

    # get the data for the desired release
    desired_release = [x for x in release_response.json() if x["tag_name"] == tag].pop()

    # build the payload for the desired release
    payload = {
        "action": "published",
        "release": desired_release,
        "repository": repo_response.json(),
    }

    # submit the payload to zenodo's api
    submit_response = requests.post(
        f"https://zenodo.org/api/hooks/receivers/github/events/?access_token={access_token}",
        json=payload,
    )

    # print the response
    print(submit_response)


def _main() -> None:
    """main runner function"""
    # parse command line arguments
    user = sys.argv[1]
    repo = sys.argv[2]
    tag = sys.argv[3]
    token = sys.argv[4]

    # submit to zenodo
    _submit(user, repo, tag, token)


# entrypoint
if __name__ == "__main__":
    _main()
