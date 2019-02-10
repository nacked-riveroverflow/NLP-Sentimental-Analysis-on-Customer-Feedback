import xlrd
from textblob import TextBlob
import re
import xlsxwriter
import solr
import os
from urllib.request import urlopen


def sentiment_analyzer():
    file_path = "C:\\Users\\S6187380\\Desktop\\higher\\topic_modelling\\"

    workbook_negative = xlsxwriter.Workbook(file_path + "nps_negative.xlsx")
    worksheet_negative = workbook_negative.add_worksheet("negative")
    worksheet_negative.write(0, 0, "Review")
    worksheet_negative.write(0, 1, "Rating")

    workbook_positive = xlsxwriter.Workbook(file_path + "nps_positive.xlsx")
    worksheet_positive = workbook_positive.add_worksheet("positive")
    worksheet_positive.write(0, 0, "Review")
    worksheet_positive.write(0, 1, "Rating")

    reader = xlrd.open_workbook(file_path + 'Copy of NPS Responses May to July 2017.xlsx')
    sh = reader.sheet_by_index(0)
    z = 1
    y = 1
    for rownum in range(1, sh.nrows):
        if TextBlob(re.sub("[^a-zA-Z]", " ", sh.row_values(rownum)[7])).sentiment.polarity < 0.0 and len(
                sh.row_values(rownum)[7]) > 20:
            worksheet_negative.write(z, 0, sh.row_values(rownum)[7])
            worksheet_negative.write(z, 1, sh.row_values(rownum)[6])
            z = z + 1
        elif len(sh.row_values(rownum)[7]) > 20:
            worksheet_positive.write(y, 0, sh.row_values(rownum)[7])
            worksheet_positive.write(y, 1, sh.row_values(rownum)[6])
            y = y + 1

        if TextBlob(re.sub("[^a-zA-Z]", " ", sh.row_values(rownum)[8])).sentiment.polarity < 0.0 and len(
                sh.row_values(rownum)[8]) > 20:
            worksheet_negative.write(z, 0, sh.row_values(rownum)[8])
            worksheet_negative.write(z, 1, sh.row_values(rownum)[6])
            z = z + 1
        elif len(sh.row_values(rownum)[8]) > 20:
            worksheet_positive.write(y, 0, sh.row_values(rownum)[8])
            worksheet_positive.write(y, 1, sh.row_values(rownum)[6])
            y = y + 1

    print("process complete")
    workbook_negative.close()
    workbook_positive.close()

def positive_cleaner():
    file_path = "C:\\Users\\S6187380\\Desktop\\higher\\topic_modelling\\"
    table_name = 'nps_positive'
    url_path = 'http://localhost:8983/solr/'
    solr_path = "C:\\Users\\S6187380\\Downloads\\solr-6.5.1\\bin"

    current_directory = os.getcwd()
    os.chdir(solr_path)
    os.system('solr delete -c ' + table_name)
    os.system('solr create -c ' + table_name)
    os.chdir(current_directory)

    reader = xlrd.open_workbook(file_path + 'nps_positive.xlsx')
    sh = reader.sheet_by_index(0)
    s = solr.Solr(url_path + table_name)
    i=1
    for rownum in range(1, sh.nrows):

        doc = dict(
            review=sh.row_values(rownum)[0],
            id_val=str(i),
            flag=0
        )
        s.add(doc)
        i=i+1
    s.commit()

    keywords=["bad","sucks","awful","horrible","poor","glitches","trouble","outrageous","lack","inconvenience","slow","unfriendly","inconvenient","hard","stupid","freeze","not happy","difficult","crazy","less fee","horrible","error","unavailable","have to","boring","instead","annoying","confusing","was down","went down","awful","has bug","fix bug","faster","quicker","reduce","rediculous","hold checl","charge less","failed","fuck","unacceptable","less fee","frustrating","reject","rejected","no more","does not work","disappointed","crashing","frustrated","hate","service charge","unable","too expensive","worse","too much","worst","Annoying","complaint","complicated","outdated","too long","confusion","ridiculous"]
    #keywords = ["not happy"]
    search_str=''
    for keyword in keywords:
        if len(keyword.split(' ')) > 1:
            keyword_str=''
            for key in keyword.split(' '):
                keyword_str=keyword_str+key+'%20'
            keyword_str=keyword_str[:-3]
        else:
            keyword_str=keyword
        search_str=search_str+'("'+keyword_str+'")%20OR%20'

    search_str=search_str[:-8]
    print(search_str)
    url_call = 'http://localhost:8983/solr/' + table_name + '/select?q=('+search_str+')&wt=python'
    print(url_call)
    conn = urlopen(url_call)
    resp = eval(conn.read())
    resp_rows = str(resp['response']['numFound'])
    print(resp_rows)
    url_string = url_call + '&rows=' + resp_rows + '&start=0'
    conn = urlopen(url_string)
    resp = eval(conn.read())

    review=[]
    id_val=[]
    flag=[]

    for doc in resp['response']['docs']:
        s.delete_query('id_val:' + str(doc['id_val'][0]))
        review.append(doc['review'][0])
        id_val.append(doc['id_val'][0])
        flag.append(1)
    s.commit()

    for i in range(0,len(id_val)):
        doc = dict(
            review=review[i],
            id_val=id_val[i],
            flag=flag[i]
        )
        s.add(doc)
    s.commit()

    flag_list=[0,1]

    for flag in flag_list:
        url_call = 'http://localhost:8983/solr/' + table_name + '/select?q=(flag:'+str(flag)+')&wt=python'
        print(url_call)
        conn = urlopen(url_call)
        resp = eval(conn.read())
        resp_rows = str(resp['response']['numFound'])
        print(resp_rows)
        url_string = url_call + '&rows=' + resp_rows + '&start=0'
        conn = urlopen(url_string)
        resp = eval(conn.read())
        if flag==1:
            workbook = xlsxwriter.Workbook(file_path + "nps_negative_filter_2.xlsx")
            worksheet = workbook.add_worksheet("negative")
            worksheet.write(0, 0, "Review")
        else:
            workbook = xlsxwriter.Workbook(file_path + "nps_positive_filter_2.xlsx")
            worksheet = workbook.add_worksheet("positive")
            worksheet.write(0, 0, "Review")
        j=1
        for doc in resp['response']['docs']:
            worksheet.write(j, 0, doc['review'][0])
            j=j+1
        workbook.close()

def data_merger():
    file_path = "C:\\Users\\S6187380\\Desktop\\higher\\topic_modelling\\"
    workbook = xlsxwriter.Workbook(file_path + "base_data_may_july.xlsx")
    worksheet = workbook.add_worksheet("negative")
    worksheet.write(0, 0, "Review")

    reader = xlrd.open_workbook(file_path + 'Copy of NPS Responses May to July 2017.xlsx')
    sh = reader.sheet_by_index(0)
    z=1
    for rownum in range(1, sh.nrows):
        if len(sh.row_values(rownum)[7]) > 20:
            worksheet.write(z, 0, sh.row_values(rownum)[7])
            z=z+1
        if len(sh.row_values(rownum)[8]) > 20:
            worksheet.write(z, 0, sh.row_values(rownum)[8])
            z = z + 1
    workbook.close()



if __name__ == '__main__':
    #sentiment_analyzer()

    #positive_cleaner()

    data_merger()
