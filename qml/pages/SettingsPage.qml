import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme
import "../components" as Components

Item {
    id: root
    objectName: "settingsPage"
    property var controller
    property var processingController
    property string projectId: ""
    property string projectName: ""
    property var pageState: ({})
    property var draft: ({})
    property var projectOverrides: ({})
    property bool dirty: false
    property string message: ""
    property string pendingMode: ""
    property var pendingSelection: null
    signal leaveRequested()
    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Typography { id: typography }
    readonly property var fields: [
        {key:"performance_mode", title:"Processing mode", options:["Normal", "Performance"], values:["normal","performance"]},
        {key:"blur_policy", title:"Blur handling", options:["Automatic","Review Borderline","Keep All"], values:["automatic","review_borderline","keep_all"]},
        {key:"face_quality_preset", title:"Face-quality sensitivity", options:["Relaxed","Balanced","Strict"], values:["relaxed","balanced","strict"]},
        {key:"keepers_per_group", title:"Preferred photographs per duplicate group", options:["1","2","3","4","5"], values:[1,2,3,4,5]},
        {key:"prefer_open_eyes", title:"Prefer open eyes", options:["On","Off"], values:[true,false]},
        {key:"prioritize_primary_face_quality", title:"Prioritize face quality", options:["On","Off"], values:[true,false]}
    ]

    function reload() {
        pageState = controller ? controller.pageData(projectId) : {values:{}, overrides:{}, globalValues:{}, defaults:{}, logicalThreads:1, physicalCores:-1, normalWorkers:2, performanceWorkers:4, performanceAvailable:false, effectiveMode:"normal"}
        draft = Object.assign({}, pageState.values)
        projectOverrides = Object.assign({}, pageState.overrides)
        dirty = false
        message = ""
    }
    function choose(key, value) {
        draft = Object.assign({}, draft, {[key]:value})
        if (projectId !== "") projectOverrides = Object.assign({}, projectOverrides, {[key]:value})
        dirty = true
    }
    function saveNow() {
        let changes = projectId !== "" ? projectOverrides : draft
        if (!controller || !controller.save(projectId, changes)) {
            message = "Could not save settings. Check the selected values."
            return
        }
        reload()
        message = "Saved. Some analysis results may need to be refreshed to reflect these settings."
    }
    function save() { saveNow() }
    function requestLeave() { if (dirty) unsavedDialog.open(); else leaveRequested() }
    Component.onCompleted: reload()
    onProjectIdChanged: reload()

    Rectangle { anchors.fill: parent; color: theme.appBackground }
    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        ColumnLayout {
            width: Math.min(parent.width, 840)
            x: Math.max(0, (parent.width-width)/2)
            spacing: 18
            Item { Layout.preferredHeight: 20 }
            Components.PillButton { text: "Back"; variant: "translucent"; onClicked: root.requestLeave() }
            Label { text: root.projectId ? "Project Settings" : "Global Settings"; font.family: typography.pageTitleFamily; font.pixelSize: 30; font.bold: true; color: theme.mainBlue }
            Label { visible: root.projectId !== ""; text: root.projectName; font.family: typography.buttonFamily; font.pixelSize: 18; color: theme.mainBlue }
            Label {
                text: root.projectId ? "Customize how Room36 processes this project. Changes here won't affect your global preferences."
                                     : "Default preferences used across all your projects."
                wrapMode: Text.WordWrap; Layout.fillWidth: true; color: theme.mutedText
            }
            Repeater {
                model: root.fields
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: content.implicitHeight + 28
                    radius: metrics.radiusLarge; color: theme.panelFill; border.color: theme.separator
                    ColumnLayout {
                        id: content
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 14
                        spacing: 8
                        Label { text: modelData.title; font.family: typography.buttonFamily; font.bold: true; color: theme.mainBlue }
                        RowLayout {
                            Layout.fillWidth: true
                            ComboBox {
                                id: selection
                                objectName: "setting_" + modelData.key
                                Layout.preferredWidth: 290
                                model: modelData.options
                                currentIndex: Math.max(0, modelData.values.indexOf(root.draft[modelData.key]))
                                onActivated: {
                                    let value = modelData.values[index]
                                    if (modelData.key === "performance_mode" && value === "performance") {
                                        if (!root.pageState.performanceAvailable) {
                                            selection.currentIndex = modelData.values.indexOf(root.draft.performance_mode)
                                            return
                                        }
                                        if (root.draft.performance_mode !== "performance") {
                                            root.pendingMode = "performance"
                                            root.pendingSelection = selection
                                            cpuWarning.open()
                                            return
                                        }
                                    }
                                    root.choose(modelData.key, value)
                                }
                                implicitHeight: metrics.toolbarButtonHeight
                                leftPadding: metrics.spacingMedium
                                rightPadding: metrics.spacingLarge
                                font.family: typography.buttonFamily
                                font.pixelSize: typography.button
                                background: Rectangle {
                                    radius: metrics.toolbarButtonRadius
                                    color: theme.panelRaisedFill
                                    border.width: selection.activeFocus || selection.popup.visible ? 2 : 1
                                    border.color: selection.activeFocus || selection.popup.visible ? theme.focusRing : theme.separator
                                }
                                contentItem: Text {
                                    text: selection.displayText
                                    color: theme.black
                                    font: selection.font
                                    verticalAlignment: Text.AlignVCenter
                                    elide: Text.ElideRight
                                }
                                indicator: Components.FluentIcon {
                                    x: selection.width - width - metrics.spacingSmall
                                    y: (selection.height - height) / 2
                                    name: "chevron_down"; size: 16; color: theme.mainBlue
                                }
                                popup: Popup {
                                    y: selection.height + 4
                                    width: selection.width
                                    implicitHeight: Math.min(contentItem.implicitHeight + 8, 280)
                                    padding: 4
                                    background: Rectangle { radius: metrics.radiusLarge; color: theme.white; border.color: theme.separator }
                                    contentItem: ListView {
                                        clip: true
                                        implicitHeight: contentHeight
                                        model: selection.popup.visible ? selection.delegateModel : null
                                        currentIndex: selection.highlightedIndex
                                        ScrollIndicator.vertical: ScrollIndicator {}
                                    }
                                }
                                delegate: ItemDelegate {
                                    width: selection.width - 8
                                    text: modelData
                                    enabled: !(selection.objectName === "setting_performance_mode" && index === 1 && !root.pageState.performanceAvailable)
                                    font.family: typography.buttonFamily
                                    highlighted: selection.highlightedIndex === index
                                    background: Rectangle {
                                        radius: metrics.toolbarButtonRadius
                                        color: parent.highlighted ? theme.selectedFill : parent.hovered ? theme.hoverFill : theme.white
                                    }
                                }
                            }
                        }
                        Label {
                            visible: modelData.key === "performance_mode"
                            text: "Detected physical cores: " + (root.pageState.physicalCores < 0 ? "unavailable" : root.pageState.physicalCores) + "; available logical threads: " + root.pageState.logicalThreads + ". Effective mode: " + (root.draft.performance_mode === "performance" && root.pageState.performanceAvailable ? "Performance" : "Normal") + ". Decoder reservation ceiling: 2 GiB."
                            wrapMode: Text.WordWrap; Layout.fillWidth: true; color: theme.mutedText
                        }
                        Label {
                            visible: modelData.key === "performance_mode" && !root.pageState.performanceAvailable
                            text: "Performance mode requires at least 4 physical CPU cores and 6 available logical threads. A saved Performance preference will run as Normal until this hardware is eligible again."
                            wrapMode: Text.WordWrap; Layout.fillWidth: true; color: theme.mutedText
                        }
                    }
                }
            }
            Label { text: root.message; visible: text.length > 0; wrapMode: Text.WordWrap; Layout.fillWidth: true; color: theme.mainBlue }
            Components.PillButton {
                text: "Refresh analysis"
                visible: root.projectId !== "" && root.message.length > 0 && root.processingController !== null
                onClicked: root.processingController.retry()
            }
            RowLayout {
                Components.PillButton { text: "Save"; variant: "accent"; enabled: root.dirty; onClicked: root.save() }
                Components.PillButton { text: "Cancel"; variant: "translucent"; onClicked: root.requestLeave() }
                Item { Layout.fillWidth: true }
                Components.PillButton { text: "Reset to Defaults"; variant: "translucent"; onClicked: resetDialog.open() }
            }
            Item { Layout.preferredHeight: 24 }
        }
    }
    Dialog {
        id: cpuWarning
        title: "Enable Performance Mode?"
        modal: true; anchors.centerIn: parent; width: 460
        background: Rectangle { radius: metrics.radiusLarge; color: theme.white; border.color: theme.separator }
        footer: DialogButtonBox {
            Components.PillButton { text: "Cancel"; variant: "translucent"; DialogButtonBox.buttonRole: DialogButtonBox.RejectRole }
            Components.PillButton { text: "Enable Performance"; variant: "accent"; DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole }
        }
        onAccepted: { root.choose("performance_mode", root.pendingMode); root.pendingMode = ""; root.pendingSelection = null }
        onRejected: {
            if (root.pendingSelection) root.pendingSelection.currentIndex = root.draft.performance_mode === "performance" ? 1 : 0
            root.pendingMode = ""
            root.pendingSelection = null
        }
        contentItem: Label { text: "Performance mode uses more CPU resources to process images faster. Your computer may become less responsive while scanning large projects."; wrapMode: Text.WordWrap; width: 420 }
    }
    Dialog {
        id: resetDialog
        title: "Reset settings?"
        modal: true; anchors.centerIn: parent; width: 460
        background: Rectangle { radius: metrics.radiusLarge; color: theme.white; border.color: theme.separator }
        footer: DialogButtonBox {
            Components.PillButton { text: "Cancel"; variant: "translucent"; DialogButtonBox.buttonRole: DialogButtonBox.RejectRole }
            Components.PillButton { text: "Reset"; variant: "accent"; DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole }
        }
        onAccepted: { if (root.controller) root.controller.reset(root.projectId); root.reload() }
        contentItem: Label { text: root.projectId ? "Remove all overrides for this project?" : "Restore global application defaults?" }
    }
    Dialog {
        id: unsavedDialog
        title: "Unsaved changes"
        modal: true; anchors.centerIn: parent; width: 460
        background: Rectangle { radius: metrics.radiusLarge; color: theme.white; border.color: theme.separator }
        contentItem: Label { text: "Save changes before leaving?" }
        footer: DialogButtonBox {
            Components.PillButton { text: "Save"; variant: "accent"; DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole; onClicked: { root.save(); if (!root.dirty) root.leaveRequested() } }
            Components.PillButton { text: "Discard"; variant: "translucent"; DialogButtonBox.buttonRole: DialogButtonBox.DestructiveRole; onClicked: root.leaveRequested() }
            Components.PillButton { text: "Cancel"; variant: "translucent"; DialogButtonBox.buttonRole: DialogButtonBox.RejectRole }
        }
    }
}


