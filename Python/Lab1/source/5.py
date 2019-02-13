'''Library Management System
Classes : Person, Book Details, Librarian, Allotment, Student'''
# Person Class
class Person():
#defined constructor
    def __init__(self,full_name,emailaddress,std_id,phn_num):
        self.fname = full_name
        self.email = emailaddress
        self.id = std_id
        self.pnum = phn_num
#prints name of a person
    def getname(self):
        print("Name : ", self.fname)
#prints phone number
    def getnum(self):
        print("Phone number : ", self.pnum)
#prints email address
    def getemailid(self):
        print("Student email address : ", self.email)
# defined private member
    def __getstd_id(self):
        print("Student ID :", self.id)

# Defined Book Details
class BookDetails():
# Defined __init__ constructor
    def __init__(self,Book_name,Book_auth,Book_genre):
        self.Bname = Book_name
        self.Bauth = Book_auth
        self.Bgenre = Book_genre
    #print book name
    def getbookname(self):
        print("Book Name :", self.Bname)
    #print author of the book
    def getbookauth(self):
        print("Author of the book :", self.Bauth)
    #print Book Genre
    def getbookgenre(self):
        print("Book Genre :", self.Bgenre)
# defined Librarian class
class Librarian():
# defined __init__constructor
    def __init__(self,lib_name,lib_id):
        self.lname = lib_name
        self.lid = lib_id
# print librarian details
    def  getlibrariandetails(self):
        print("Librarian details:", self.lname,self.lid)

#inheriting Librarian class
class Allotment(Librarian):
    #defined __init__ constructor
    def __init__(self, issued_date,return_date,lib_name,lib_id):
        Librarian.__init__(self,lib_name,lib_id)
        self.idate = issued_date
        self.rdate = return_date
        self.libname = lib_name
        self.libid = lib_id
    # print Book isssued date
    def getissueddate(self):
        print("Book issued on :",self.idate)
    # print Book returned date
    def getreturndate(self):
        print("Book should be returned on :",self.rdate)

# Multiple inheritance, Student Class inherits Person,BookDetails,Allotment classes
class Student(Person,BookDetails,Allotment):
    def __init__(self,full_name,emailaddress,std_id,phn_num,Book_name,Book_auth,Book_genre,
                 issued_date,return_date,lib_name,lib_id):
        super().__init__(full_name,emailaddress,std_id,phn_num)
        Allotment.__init__(self,issued_date,return_date,lib_name,lib_id)
        BookDetails.__init__(self,Book_name,Book_auth,Book_genre)

b = Student("Sushma Manne","manne@gmail.com","99999991","999-999-0001","Adventures","Charles Dickens",
            "Literature","02-12-2019","03-12-2019","Charles","11154")
print("Student Details : ")
b.getname()
b.getemailid()
b.getnum()
print("-------------------")
print("Book Details: ")
b.getbookname()
b.getbookauth()
b.getbookgenre()
print("-------------------")
print("Book Issuing Details : ")
b.getissueddate()
b.getreturndate()

#creating instances of all classes and calling their member functions.
c = Allotment("02-14-2019","03-14-2019","John","16234")
print("--------------------")
c.getreturndate()
print("--------------------")
d = Librarian("Jack,","15342")
d.getlibrariandetails()
print("--------------------")
e = Person("Bhavana Kota", "bhavz@gmail.com","16274561","999-919-9009")
e.getname()


