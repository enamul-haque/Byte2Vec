# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:59:18 2018
@author: Md. Enamul Haque
@inistitute: University of Louisiana at Lafayette
"""

import requests 
from bs4 import BeautifulSoup 
  
''' 
URL of the archive web-page which provides link to 
all folders. It would have been tiring to 
download each files from every folder manually. 
In this example, we first crawl the webpage to extract 
all the links and then download files. 
'''
  
# specify the URL of the archive here 
archive_url = "http://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/"
  
def get_links(): 
      
    # create response object 
    r = requests.get(archive_url) 
      
    # create beautiful-soup object 
    soup = BeautifulSoup(r.content,'html5lib') 
      
    # find all links on web-page 
    links = soup.findAll('a') 
  
    # filter the link sending with .mp4 
    links = [archive_url + link['href'] for link in links if link['href'].endswith('zip')] 
  
    return links 
  
  
def download_series(links): 
  
    for link in links: 
  
        '''iterate through all links in links 
        and download them one by one'''
          
        # obtain filename by splitting url and getting  
        # last string 
        file_name = link.split('/')[-1]    
  
        print ("Downloading file:%s"%file_name) 
          
        # create response object 
        r = requests.get(link, stream = True) 
          
        # download started 
        with open(file_name, 'wb') as f: 
            for chunk in r.iter_content(chunk_size = 1024*1024): 
                if chunk: 
                    f.write(chunk) 
          
        print ("%s downloaded!\n"%file_name)
  
    print ("All files downloaded!")
    
    return
    
if __name__ == "__main__": 
  
    # getting all video links 
    links = get_links() 
  
    # download all videos 
    download_series(links) 
