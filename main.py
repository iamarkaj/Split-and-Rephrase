import parser
import testT5
import filter

input_sentence = "When Prime Minister Scott Morrison publicly lambasted US car giant General Motors after it announced it was pulling the Holden brand from the Australian market, it set the tone for an ugly brawl."

x = parser.main(input_sentence)
y = testT5.main(x)
z = filter.main(x, y)
