#数据抓取

#2.1分析网页

#2.2三种网页抓取方法
#正则表达式
import urllib
import re
def download(url,user_agent='wswp',num_retries = 2):
    headers = {"User_agent":user_agent}
    request = urllib.request.Request(url,headers = headers)
    try:
        html = str(urllib.request.urlopen(request).read(),'utf-8')
    except urllib.error.URLError as e:
        print("Download error",e.reason)
        html = None
        if num_retries > 2:
            if hasattr(e,"code") and 500 <= e.code < 600:
                return download(url,num_retries - 1)


def scraping():
    url = 'https://www.zhihu.com'
    html = download(url)
    #正则过程
    re.findall('',html)

#beautiful soup
from bs4 import BeautifulSoup
def beautiful_soup():
    broken_html = '<ul class=country><li>Area<li>Pupulation</ul>'
    #parse the html
    soup = BeautifulSoup(broken_html,'html.parser')
    fixed_html = soup.prettify()
    print(fixed_html)
    ul = soup.find('ul',attrs={'class':'country'})
    li = ul.find('li')
    print(li)
    lis = ul.find_all('li')
    print(lis)

#Lxml
import lxml.html

def lxml_html():
    broken_url = '<ul class=country><li>Area<li>Pupulation</ul>'
    tree = lxml.html.fromstring(broken_url)#parse the HTML
    # tree.cssselect()
    fixed_html = lxml.html.tostring(tree, pretty_print=True)
    print(fixed_html)
lxml_html()

FIELDS = ('area', 'population', 'iso', 'country', 'capital', 'continent', 'tld',
          'currency_code', 'currency_name', 'phone', 'postal_code_format',
          'postal_code_regex', 'languages', 'neighbours')

def re_scraper(html):
    results = {}
    for field in FIELDS:
        results[field] = re.search()