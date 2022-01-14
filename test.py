from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree


def __indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


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

    root = ElementTree(books)

    __indent(books)
    root.write("test.xml", encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    main()
