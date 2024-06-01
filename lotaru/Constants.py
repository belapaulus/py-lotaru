from os import path
WORKFLOWS = {
    "eager":        ['fastqc', 'adapter_removal', 'fastqc_after_clipping',
                     'bwa', 'samtools_flagstat', 'samtools_filter',
                     'samtools_flagstat_after_filter', 'markduplicates',
                     'preseq', 'damageprofiler', 'qualimap', 'genotyping_hc',
                     'bcftools_stats'],
    "methylseq":    ['bismark_align', 'bismark_deduplicate',
                     'bismark_methxtract', 'bismark_report', 'fastqc',
                     'qualimap', 'trim_galore'],
    "chipseq":      ['bigwig', 'bwa_mem', 'fastqc', 'macs2', 'macs2_annotate',
                     'merged_bam', 'merged_bam_filter', 'phantompeakqualtools',
                     'picard_metrics', 'plotfingerprint', 'plotprofile',
                     'preseq', 'sort_bam', 'trimgalore'],
    "atacseq":      ['bwa_mem', 'fastqc', 'merged_lib_ataqv', 'merged_lib_bam',
                     'merged_lib_bam_filter', 'merged_lib_bam_remove_orphan',
                     'merged_lib_bigwig', 'merged_lib_macs2',
                     'merged_lib_macs2_annotate', 'merged_lib_picard_metrics',
                     'merged_lib_plotfingerprint', 'merged_lib_plotprofile',
                     'sort_bam', 'trimgalore'],
    "bacass":       ['nfcore_bacass:bacass:fastqc',
                     'nfcore_bacass:bacass:kraken2',
                     'nfcore_bacass:bacass:prokka',
                     'nfcore_bacass:bacass:skewer',
                     'nfcore_bacass:bacass:unicycler'],
}
NODES = ["asok01", "asok02", "n1", "n2", "c2", "local"]
TRACE_DIR = path.join("data", "traces")
BENCH_DIR = path.join("data", "benchmarks")
LOTARU_G_BENCH = path.join(BENCH_DIR, "lotaru-g.csv")
LOTARU_A_BENCH = path.join(BENCH_DIR, "lotaru-a.csv")
