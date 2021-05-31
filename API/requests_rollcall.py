import requests
from datetime import datetime
import uuid


baseUrl = 'https://localhost:44319/api/RollCalls/'
# Call request tạo mới bản ghi điểm danh
def CreateRollCall(employeeCode):
    now = datetime.now()
    day = str(now.day)
    month = str(now.month)
    year = str(now.year)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    if len(day) < 2 :
        day = "0"+ day
    if len(month) < 2 :
        month = "0"+ month
    if len(hour) < 2 :
        hour = "0"+ hour
    if len(minute) < 2 :
        minute = "0"+ minute
    if len(second) < 2 :
        second = "0"+ second

    timeCode = day + month + year + hour + minute + second

    url =   baseUrl + "CreateRollCallFromFaceRecognizeByEmployeeCode?employeeCode=" + str(employeeCode) + "&RollCallTimeCode="+ timeCode
        

    # request của mình chỉ cần gửi lên employeeCode -> Solved(lấy được thông tin ở label) và timeCode ->TODO

    createResponse = requests.post(url, verify=False)
