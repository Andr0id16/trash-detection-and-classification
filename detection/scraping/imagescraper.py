import threading
import requests
from serpapi import GoogleSearch

apikey = "129d3278283dcdf79137d82c7ea9f9e37d1e02fb922668390e79e1ca362a601b"
# for installation and usage
# https://github.com/serpapi/google-search-results-python


# CHOOSE WASTE CATEGORY
# images are stored in folder with same name as category
# NOTE : create folder of required category before running script
category = "glass"


# CREATE SEARCH QUERY
# search based on following string and paramters
query_string = "glass bottles on ground broken"
pagenumber = 1  # result page number
maxnumber = 200  # maximum number of images
params = {
    "engine": "google",
    "q": query_string,
    "api_key": apikey,
    "num": maxnumber,
    "tbm": "isch",
    "ijn": pagenumber
    # "tbs" : for advanced search
}
# for more details
# https://serpapi.com/search-api
# https://serpapi.com/images-results


# GETTING LINKS OF IMAGES FROM PARAMS
imagelinks = []
search = GoogleSearch(params)
results = search.get_dict()
images_results = results["images_results"]
for i in images_results:
    imagelinks.append(i["original"])


# DOWNLOAD AND WRITE IMAGES FROM LINKS


def writeimage(index, i):
    # requests libary is used to make GET requests on the image links
    # reponse in bytes is written to appropriately named empty binary file
    # some images take too long to be downloaded and written (timeout)
    # some images cannot be converted to jpg format (conversion error)
    # Exception handling and threads are used for handling such cases
    try:
        r = requests.get(i)
        if r.status_code == 200:
            with open(
                f"{category}\car_{category}_{pagenumber}_{index}.jpg", "wb"
            ) as image:
                image.write(r.content)
            print(f"image {index} written")
    except Exception as e:
        print(f"Error: {e} for image {index}")


# threads allow each image download to be handled individually
# each thread runs writeimage for seperate link
threads = []
for index, i in enumerate(imagelinks):
    thread = threading.Thread(target=writeimage, args=(index, i))
    threads.append(thread)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(5000)  # threads remain alive for max 5 seconds
    if thread.is_alive():
        print("thread is not done, setting event to kill thread.")
        thread.set()
    else:
        print("thread has already finished.")
