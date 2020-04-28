PATH = "../Datasets/"


def read_data():
    book_content = {}
    with open(PATH + "goblet_book.txt") as f:
        while True:
            c = f.read(1).lower()  # Read one character and convert to lower case
            if c not in book_content:
                book_content[c] = 1  # Initialize character with one repetition
            else:
                book_content[c] += 1  # Add one repetition to the character frequency
            if not c:
                print("End of file")
                break

    book_chars = [k for k in book_content.keys()]  # Save the unique characters found

    return book_chars


def main():
    char_to_ind = {}
    ind_to_char = {}
    book_chars = read_data()

    for idx, x in enumerate(book_chars):
        char_to_ind[x] = idx
        ind_to_char[idx] = x


if __name__ == "__main__":
    main()
