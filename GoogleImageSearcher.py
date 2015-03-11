from threading import Thread
from Queue import Queue
import shutil
import requests
import os
import re
import copy
from datetime import datetime
from selenium import webdriver
import random
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
import sys
import os.path
import time
import urlparse
from PIL import Image
import io
from urllib import urlencode
from urlparse import parse_qs, urlsplit, urlunsplit, urlparse
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

phantom_path="/path/to/phantomjs"

class GoogleImageSearcher(object):

    def __init__(self):

        self.google_urls = ['https://www.google.com.au/imghp', 'https://www.google.com/imghp']
        self.google_search_urls = ['http://www.google.com.au/search', 'http://www.google.com/search']
        self.concurrent = 2
        dcap = dict(webdriver.DesiredCapabilities.PHANTOMJS)
        dcap["phantomjs.page.settings.userAgent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.111 Safari/537.36")
        self.que = Queue(self.concurrent * 2)
        for i in range(self.concurrent):
           t = Thread(target=self.doWork)
           t.daemon = True
           t.start()
    #  Assigning the user agent string for PhantomJS
        self.browser = webdriver.PhantomJS(executable_path=phantom_path,
                                      desired_capabilities=dcap)
        self.browser.implicitly_wait(60)

    def addQueryParameter(self, url, param_name, param_value):
        scheme, netloc, path, query_string, fragment = urlsplit(url)
        query_params = parse_qs(query_string)

        query_params[param_name] = [param_value]
        new_query_string = urlencode(query_params, doseq=True)
        return urlunsplit((scheme, netloc, path, new_query_string, fragment))

    def isValidImage(self, imageName, dimension):
        try:
            im = Image.open(imageName)
            bands = im.getbands()
            if (len(bands) == 1):
                print " Got a black and white image %s"%(imageName)
                return -1
            if (dimension != (0,0)):
                #print("resizing "+str(dimension) + " for "+ imageName)
                im = im.resize(dimension)
                im.save(imageName)
            return 1
        except Exception, e:
            print "Couldn't check if %s is a valid image: %s" % (imageName, e)
            return -1


    def getImagesByKeyword(self, keywords, outputFolder="output", numOfImagesWanted=5):
        similar_image_links = self.searchfileByKeyword(keywords, numOfImagesWanted * 10)
        success = 0
        for url in similar_image_links:
            try:
                if (self.download(url, outputFolder) == 0):
                    success = success + 1
                if (success == numOfImagesWanted):
                    break
            except Exception,e :
                print "Exception when downloading %s : %s" % (url, e)
                continue

            #self.que.put([url, outputFolder])
        #self.que.join()

    def getImages(self, inputImageFilename, outputFolder="output", numOfImagesWanted = 5, dimension=(0,0)):
        print (" **************** Searching for Images in Google "+ str(numOfImagesWanted * 10))
        similar_image_links = self.searchfile(inputImageFilename, numOfImagesWanted * 10)
        success = 0
        print (" **************** Downloading for Images in Google "+ str(numOfImagesWanted*10))
        for url in similar_image_links:
            if (self.download(url, outputFolder, dimension) == 0):
                success = success + 1
            if (success == numOfImagesWanted*10):
                break
            #self.que.put([url, outputFolder])
        #self.que.join()

    def doWork(self):
        while True:
            [url, outputFolder] = self.que.get()
            self.download(url, outputFolder)
            self.que.task_done()

    def download(self, url, outputFolder, dimension):
        try:
            response = requests.get(url, stream=True)
            contentType = response.headers['Content-Type']
        except Exception, e:
            print "Couldn't download url %s : %s" % (url, e)
            return -1

        if (response.status_code != 200 or (contentType != "image/jpeg" and contentType != "image/jpg")):
            #print(" Got response code "+ str(response.status_code)  + " content type "+contentType + " for "+url)
            return -1;
        path = copy.copy(url)
        o  = urlparse(path)
        path = o.netloc + o.path
        path = re.sub("[/:\?\.]","-", path)
        path = path[0:127];
        path = path + ".jpg"
        #file = os.path.basename(path)
        fullpath = outputFolder + "/"+path
        ret = -1
        with open(fullpath, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
            if (self.isValidImage(fullpath, dimension) == -1):
                os.remove(fullpath)
                #print ("removing bad image " + fullpath + " for url "+url)
                ret = -1
            else:
                #print ("CREATING FILE " + path + " for url "+ url)
                ret = 0
        del response
        return ret


    def searchfileByKeyword(self, googleQuery, numOfImagesWanted):
        similar_image_links = []
        print ("Image Search Query " + googleQuery)
        url = random.choice(self.google_search_urls)
        url = self.addQueryParameter(url, "q", googleQuery)
        self.browser.get(url)
        self.browser.implicitly_wait(20)
        #Image.open(io.BytesIO(self.browser.get_screenshot_as_png())).show()
        #input = self.browser.find_elements_by_xpath("//input[@id='q']")
        #Image.open(io.BytesIO(self.browser.get_screenshot_as_png())).show()
        elemImage = self.browser.find_element_by_link_text("Images")
        elemImage.click()
        ele2 = self.browser.find_elements_by_xpath("//div[@data-ri]//a")
        got = 0
        for element in ele2:
            similar_image_links.append(self.get_real_image_url(element.get_attribute('href')))
            got = got+1
            if (got == numOfImagesWanted):
                break
        print("similar_image_links= ")
        return similar_image_links


#  Function to search file from local machine
    def searchfile(self, inputFileName, numOfImagesWanted):
        similar_image_links = []
        url = random.choice(self.google_urls)
        print "Connecting to Google Image Search " + url
        self.browser.get(url)
        self.browser.implicitly_wait(20)
        # Click "Search by image" icon
        elem = self.browser.find_element_by_class_name('gsst_a')
        elem.click()
        # Switch from "Paste image URL" to "Upload an image"
        self.browser.execute_script("google.qb.ti(true);return false")
        # Set the path of the local file and submit
        print "Uploading file to 'Search by Image'"
        elem = self.browser.find_element_by_id("qbfile")
        elem.send_keys(inputFileName)
        #input = self.browser.find_elements_by_xpath("//input[@type='submit']")
        #input[0].click()
        #Clicking 'Visually Similar Images'
        self.browser.implicitly_wait(20)
        print ("Searching for most similar match")
        ele1 = self.browser.find_element_by_link_text("Visually similar images")
        url = ele1.get_attribute('href')
        googleQuery = self.getGoogleQuery(url)
        googleQuery = ""
        if (googleQuery != ""):
            googleQuery = googleQuery + " color image photo "
            return self.searchfileByKeyword(googleQuery, numOfImagesWanted)
        else:
            print("Could not Map the Image to key words")
            ele1.click()

        ele2 = self.browser.find_elements_by_xpath("//div[@data-ri]//a")
        got = 0
        print("numOfImagesWanted = "+str(numOfImagesWanted))
        for element in ele2:
            similar_image_links.append(self.get_real_image_url(element.get_attribute('href')))
            got = got+1
            if (got == numOfImagesWanted):
               break
        #print("similar_image_links= "+str(similar_image_links))
        return similar_image_links

    def get_real_image_url(self, image_url):
        parts = urlparse(image_url)
        query = parse_qs(parts.query, keep_blank_values=True)
        return query.get("imgurl")[0]

    def getGoogleQuery(self, imageUrl):
        parts = urlparse(imageUrl)
        queries = parse_qs(parts.query, keep_blank_values = True)
        query = queries.get("q")
        if query is not None:
            return query[0]
        return ""

if __name__ == '__main__':
    filename="../input/bwlittlegirl.jpeg"
    gis = GoogleImageSearcher()
    #gis.getImages(filename, "internetimages", 50, (190,266))
    gis.getImages(filename, "internetimages", 50, (0,0))
    ##gis.searchfile2(filename, 5)
    #gis.getImagesByKeyword("sad indian girl", "output", 5)


