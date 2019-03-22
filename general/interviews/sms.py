# SMS splitter

SMS_LENGTH = 100

def text_append(i, n):
    return "[ Part %d of %d ] "%(i,n)

def insert_at(index, text, insertion):
    return text[:index] + insertion + text[index:]

def main():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur"
    a = text_append(1,1)
    l = len(a)
    tl = len(text)
    if len(text) > SMS_LENGTH:

        total_length = tl
        i = 1
        while i*SMS_LENGTH <= total_length: 
            i = i + 1
            total_length = l*i + tl
        print("Total parts", i)
        sms_parts = i
        smses = []
        for part in range(1, sms_parts+1):
            begin = SMS_LENGTH*(part-1)
            end = begin + SMS_LENGTH
            sms = insert_at(begin, text ,text_append(part, sms_parts))
            smses.append(sms[begin: end])
        print(smses)



if __name__ == "__main__":
    main()