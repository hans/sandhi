
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

from tqdm import tqdm


wildcard_constraints:
    speaker = "SSB[0-9]{4}"


# corpus_speakers, corpus_files = glob_wildcards("data/AISHELL/train/wav/{speaker}/{file}.wav")
corpus_speakers, _ = glob_wildcards("data/AISHELL/train/wav/{speaker}/{file}.wav")
corpus_speakers = list(set(corpus_speakers))


rule preprocess_transcript:
    input:
        text = "data/AISHELL/train/content.txt"
    output:
        # expand("intermediates/AISHELL/transcript/hanzi/{file}.txt", file=corpus_files),
        # expand("intermediates/AISHELL/transcript/pinyin/{file}.txt", file=corpus_files)
        directory("intermediates/AISHELL/transcript")
    run:
        shell("mkdir -p intermediates/AISHELL/transcript/hanzi")
        shell("mkdir -p intermediates/AISHELL/transcript/pinyin")

        with open(input.text, "r") as f:
            num_lines = sum(1 for line in f)
        with open(input.text, "r") as f:
            for line in tqdm(f, total=num_lines):
                line = line.strip()
                identifier, text = line.split(maxsplit=1)
                identifier = identifier.replace(".wav", "")

                with open(f"intermediates/AISHELL/transcript/hanzi/{identifier}.txt", "w") as f_hanzi, \
                     open(f"intermediates/AISHELL/transcript/pinyin/{identifier}.txt", "w") as f_pinyin:
                    for i, token in enumerate(text.split(" ")):
                        if i % 2 == 0:
                            f_hanzi.write(token)
                            f_hanzi.write(" ")
                        else:
                            f_pinyin.write(token)
                            f_pinyin.write(" ")

rule preprocess_audio:
    input:
        rules.preprocess_transcript.output,
        audio = "data/AISHELL/train/wav",
        text = "intermediates/AISHELL/transcript/hanzi"
    output:
        directory("intermediates/AISHELL/mfa_input")
    shell:
        """
        mkdir -p {output}
        for speaker in {corpus_speakers}; do
            mkdir -p {output}/$speaker
            for file in {input.audio}/$speaker/*.wav; do
                outpath=$(grealpath --relative-to={input.audio} $file)
                ffmpeg -v 24 -stats -i $file -ac 1 -ar 16000 {output}/$outpath

                cp {input.text}/$(basename $file .wav).txt {output}/$(echo $outpath | sed "s/.wav/.txt/")
            done
        done
        """


rule generate_mfa_dictionary:
    input:
        corpus = directory("intermediates/AISHELL/mfa_input"),
        g2p_model = HTTP.remote("https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/mandarin_character_g2p.zip")
    conda: "envs/mfa.yaml"
    output:
        "intermediates/AISHELL/dictionary.txt"
    shell:
        "mfa g2p {input.g2p_model} {input.corpus} {output}"


rule align_audio:
    input:
        corpus = directory("intermediates/AISHELL/mfa_input"),
        dictionary = "intermediates/AISHELL/dictionary.txt",

        pretrained_model = HTTP.remote("https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-mandarin_mfa-v2.0.0a/mandarin_mfa.zip")
    conda: "envs/mfa.yaml"
    output:
        directory("intermediates/AISHELL/aligned")
    shell:
        """
        mfa align --clean {input.corpus} {input.dictionary} {input.pretrained_model} {output}
        """