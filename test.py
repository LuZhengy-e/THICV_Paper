from xml.etree.ElementTree import Element, SubElement, tostring


def main():
    BOOKs = {
        '001': {
            'title': 'Name1',
            'edition': 2,
            'year': 2006,
        },
        '002': {
            'title': 'Name2',
            'year': 2009,
        },
    }
    books = Element('books')
    for isbn, info in BOOKs.items():  # 此处若用python2，则改为iteritems()
        book = SubElement(books, 'book')
        info.setdefault('authors', 'help')
        info.setdefault('edition', 1)
        for key, val in info.items():
            SubElement(book, key).text = ', '.join(str(val).split(':'))
    xml = tostring(books)
    print('*** RAW XML ***')
    print(xml)


if __name__ == '__main__':
    main()
