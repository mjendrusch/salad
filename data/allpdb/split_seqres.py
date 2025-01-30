import sys
if __name__=="__main__":
    seqres = sys.argv[1]
    basename = ".".join(seqres.split(".")[:-1])
    with open(seqres) as f, open(f"{basename}.na.txt", "w") as na, open(f"{basename}.aa.txt", "w") as aa:
        lines = iter(f)
        while True:
            try:
                header = next(lines)
                sequence = next(lines)
                is_na = "mol:na" in header
                is_aa = "mol:protein" in header
                if is_na:
                    na.write(header)
                    na.write(sequence)
                if is_aa:
                    aa.write(header)
                    aa.write(sequence)
            except StopIteration:
                break
