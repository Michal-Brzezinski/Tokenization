import textdistance

s1 = "Ala ma kota"
s2 = "Ala posiada kota"

distance = textdistance.levenshtein.distance(s1, s2)

print("Odległość Levenshteina:", distance)