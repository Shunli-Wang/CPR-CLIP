import random
import clip
import torch
from itertools import combinations

actionList = [
    'Correct', 
    'Overlap Hands', 
    'Clenching Hands', 
    'Single Hand', 
    'Bending Arms', 
    'Tilting Arms', 
    'Jump Pressing', 
    'Squatting', 
    'Standing', 
    'Wrong Position', 
    'Insufficient Pressing', 
    'Slow Frequency', 
    'Excessive Pressing', 
    'Random Position Pressing'
]
adviceList = [
    '', 
    'Cross hands', 
    'Cross hands', 
    'Using both hands to press', 
    'Keep arms straight', 
    'Keep arms vertical', 
    'Stop Jumping Press', 
    'Adopting a kneeling position', 
    'Adopting a kneeling position', 
    'Adjusting the pressing position', 
    'Increase the number of presses', 
    'Increase the compression frequency', 
    'Reduce compression amplitude', 
    'Maintain the pressed position'
]

# Counts
cntTxtList = [
        f"This clip contains no errors.",
        f"This clip contains one error.", 
        f"This clip contains two errors.",
        f"This clip contains three errors.", 
        f"This clip contains four errors."
        ]

# Classes
errClsTxt1 = [
            f"Correct CPR action.",
            f"This subject made a mistake of {{}}.", 
            f"This subject made both {{}} and {{}} mistakes.", # , simultaneously
            f"This subject made {{}}, {{}} and {{}} mistakes.", # , simultaneously
            f"This subject made {{}}, {{}}, {{}}, and {{}} mistakes.", # , simultaneously
            ]
errClsTxt2 = [
            f"Correct CPR action.",
            f"Error in this clip: {{}}.", 
            f"Errors in this clip: {{}} and {{}}.", 
            f"Errors in this clip: {{}}, {{}} and {{}}.", 
            f"Errors in this clip: {{}}, {{}}, {{}} and {{}}.", 
            ]

# Advise
advTxt1 = [
        f"Already good enough, no suggestions for improvement.",
        f"The subject should {{}}.", 
        f"The subject should {{}}, and {{}}.", 
        f"The subject should {{}}, {{}}, and {{}}.", 
        f"The subject should {{}}, {{}}, {{}}, and {{}}.", 
        ]
advTxt2 = [
        f"Already good enough, no suggestions for improvement.",
        f"Improvement points: {{}}.", 
        f"Improvement points: {{}}, and {{}}.", 
        f"Improvement points: {{}}, {{}}, and {{}}.", 
        f"Improvement points: {{}}, {{}}, {{}}, and {{}}.", 
        ]

def getPrompt(idList, setFlag=False):
    # ptr 
    if setFlag is True:
        advTxtPtr = advTxt1
        errClsTxtPtr = errClsTxt1
    else:
        advTxtPtr = random.sample([advTxt1, advTxt1], 1)[0]
        errClsTxtPtr = random.sample([errClsTxt1, errClsTxt2], 1)[0]

    cntTxt, actTxt = None, None
    errCls, advTxt = None, None

    # [0] 
    if len(idList) == 1 and idList[0] == 0: 
        cntTxt = cntTxtList[0] # "This clip contains no errors.",
        advTxt = adviceList[0] # ''
        errCls = errClsTxtPtr[0] # "Correct CPR action."
        advTxt = advTxtPtr[0] # "Already good enough, no suggestions for improvement."
        return cntTxt, errCls, advTxt

    # Others
    cntTxt = cntTxtList[len(idList)] # "This clip contains only one error."
    actTxt = [actionList[id] for id in idList] # 
    advTxt = [adviceList[id] for id in idList] # 

    errCls = errClsTxtPtr[len(idList)].format(*actTxt)
    advTxt = advTxtPtr[len(idList)].format(*advTxt)

    return cntTxt, errCls, advTxt

def tokenTestLabel():
    # Get prompt and tokens in testing
    combin1 = [[i] for i in range(0, 14)] # 
    promptTknList = []
    for idList in combin1:
        cntTxt, errCls, advTxt = getPrompt(idList, setFlag=True)
        txtFeat = clip.tokenize(cntTxt + errCls + advTxt).cuda() # All
        # txtFeat = clip.tokenize(errCls + advTxt).cuda() # wo/ cntTxt
        # txtFeat = clip.tokenize(cntTxt + advTxt).cuda() # wo/ errCls
        # txtFeat = clip.tokenize(cntTxt + errCls).cuda() # wo/ advTxt
        promptTknList.append(txtFeat)
    promptTkn = torch.cat(promptTknList) # (8, 77) 

    return promptTkn, combin1

def tokenBatchLabel(label):
    '''Generate tokens of '''
    labelID = [list(torch.nonzero(i).flatten().long().numpy()) for i in label] # onehot to id list. [[1,2], [3,4]] 
    promptTknList = []
    for idList in labelID:
        cntTxt, errCls, advTxt = getPrompt(idList)
        txtFeat = clip.tokenize(cntTxt + errCls + advTxt).cuda()
        promptTknList.append(txtFeat)
    promptTkn = torch.cat(promptTknList) # (8, 77)

    return promptTkn, labelID 


if __name__ == '__main__':
    idList = [
        [0],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        ]

    cntTxt, errCls, advTxt = getPrompt([1, 2, 3, 4]) 

    print(cntTxt)
    print(errCls)
    print(advTxt)

    allClsPrompt, _ = getTestPrompt()
    print(allClsPrompt.shape)
