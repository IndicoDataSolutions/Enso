import traceback

def get_rationales_by_sentences(sentences, rationales):
        rationales_by_sentences = [[]]
        rationales = sorted(rationales, key=lambda r: r['start'])
        text_idx = len(sentences[0]) + 2
        s_idx = 1
        for rationale in rationales:
            if s_idx > len(sentences):
                break

            if rationale['start'] < text_idx:
                rationales_by_sentences[-1].append(rationale)
            else:
                text_idx += len(sentences[s_idx]) + 2
                s_idx += 1
                while rationale['start'] >= text_idx:
                    text_idx += len(sentences[s_idx]) + 2
                    s_idx += 1
                    rationales_by_sentences.append([])
                rationales_by_sentences.append([rationale])
        # flush
        while s_idx < len(sentences):
            s_idx += 1
            rationales_by_sentences.append([])
        return rationales_by_sentences