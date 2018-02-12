
#识别网站所用的技术
import builtwith as bw

def builtwithParse():
    print(bw.parse('https://www.zhihu.com'))

# builtwithParse()
#下载网页
import urllib

# def download(url):
#     return urllib.request.urlopen(url).read()
# def download(url):
#     print("Downloading:",url)
#     try:
#         html = urllib.request.urlopen(url).read()
#     except urllib.error.URLError as e:
#         print("Download error:",e.reason)
#         html = None
#     return html

# def download(url,num_retries=2):
#     print("Downloading:",url)
#     try:
#         html = urllib.request.urlopen(url).read()
#     except urllib.error.URLError as e:
#         print("Download error:",e.reason)
#         html = None
#         if num_retries > 2:
#             if hasattr(e,'code') and 500 <= e.code < 600:
#                 return download(url,num_retries - 1)
#     return html

def download(url,user_agent='wswp',num_retries=2):
    print("Downloading:",url)
    headers = {"User-agent":user_agent}
    request = urllib.request.Request(url,headers = headers)
    try:
        html = str(urllib.request.urlopen(request).read(),'utf-8')
    except urllib.error.URLError as e:
        print("Download error:",e.reason)
        html = None
        if num_retries > 2:
            if hasattr(e,'code') and 500 <= e.code < 600:
                return download(url,num_retries - 1)
    return html

# print(download('http://www.xidian.edu.cn'))

import re

# 网站地图爬虫
# def crawl_sitemap(url):
#     # download the sitmap file
#     sitemap = download(url)
#     #extract the sitemap links
#     links = re.findall('<a(.*)>(.*)</a>',str(sitemap))
#     #download each link
#     for link in links:
#         print(link)
#         html = download(link)
#         #scrape html here
#         #...

#coding=utf8
# def crawl_sitemap(url):
#     # download the sitmap file
#     sitemap = download(url)
#     #extract the sitemap links
#     spans = re.findall('<span(.*)>(.*)</span>',str(sitemap))
#     #download each link
#     for span in spans:
#
#         print(span)
#         #scrape html here
#         #...
#
# crawl_sitemap("https://www.zhihu.com")

#id 遍历爬虫

#链接爬虫
def link_crawler(seed_url, link_regex):
    """Crawl from the given seed URL following links matched by link_regex"""

    crawl_queue = [seed_url]
    while crawl_queue:
        url = crawl_queue.pop()
        html = download(url)
        #filter for links matching our regular expression
        for link in get_link(html):
            if re.match(link_regex,link):
                crawl_queue.append(link)

def get_link(html):
    """Return a list of links from html"""
    # a regular expression to extract all links from the webpage


#高级功能

import urllib.robotparser as rp
def robot():
    mrp = rp.RobotFileParser()
    mrp.set_url('https://www.tmall.com/robots.txt')
    mrp.read()
    url = 'https://www.baidu.com'
    user_agent = 'BadCrawler'
    flag = mrp.can_fetch(user_agent,url)
    print(flag)
    user_agent = ''
    flag = mrp.can_fetch(user_agent, url)
    print(flag)


# robot()

