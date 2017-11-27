from time import sleep
import json
import urllib2

myAPIstr = 'AIzaSyDsvOXMUNyk96jiq3W6kfsVzTxfrDxS6Mk'
setNum = 8288
searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % \
            (myAPIstr, setNum)
pg = urllib2.urlopen(searchURL)